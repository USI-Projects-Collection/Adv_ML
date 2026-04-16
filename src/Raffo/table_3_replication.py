import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.datasets import VOCDetection
from PIL import Image
import numpy as np
import scipy.ndimage as ndimage
from tqdm import tqdm
import timm
import open_clip 
from transformers import AutoModel

# ==========================================
# 1. FEATURE EXTRACTOR (From your previous code)
# ==========================================
class AttentionFeatureExtractor:
    def __init__(self, model, feature_type="keys"):
        self.model = model
        self.feature_type = feature_type.lower()
        self.features = None
        self.hook_handle = None
        self.model_backend = None
        self._register_hook()

    def _register_hook(self):
        if hasattr(self.model, 'blocks'): 
            last_attn_layer = self.model.blocks[-1].attn
            self.model_backend = "timm"
        elif hasattr(self.model, 'visual') and hasattr(self.model.visual, 'transformer'): 
            last_attn_layer = self.model.visual.transformer.resblocks[-1].attn
            self.model_backend = "open_clip"
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'visual') and hasattr(self.model.model.visual, 'transformer'):
            last_attn_layer = self.model.model.visual.transformer.resblocks[-1].attn
            self.model_backend = "open_clip"
        elif hasattr(self.model, 'vision_model'): 
            last_attn_layer = self.model.vision_model.encoder.layers[-1].self_attn
            self.model_backend = "hf_clip"
        else:
            raise NotImplementedError("Could not identify the model architecture.")
        
        def hook(module, input, output):
            x = input[0]
            B, N, C = x.shape
            if self.model_backend == "timm":
                qkv = module.qkv(x)
                qkv = qkv.reshape(B, N, 3, module.num_heads, C // module.num_heads).permute(2, 0, 3, 1, 4)
                feat = qkv[1] if self.feature_type == "keys" else qkv[2]
                self.features = feat.transpose(1, 2).reshape(B, N, C)
            elif self.model_backend == "open_clip":
                qkv = F.linear(x, module.in_proj_weight, module.in_proj_bias)
                qkv = qkv.reshape(B, N, 3, module.num_heads, C // module.num_heads).permute(2, 0, 3, 1, 4)
                feat = qkv[1] if self.feature_type == "keys" else qkv[2]
                self.features = feat.transpose(1, 2).reshape(B, N, C)
            elif self.model_backend == "hf_clip":
                if self.feature_type == "keys":
                    feat = module.k_proj(x)
                else:
                    feat = module.v_proj(x)
                self.features = feat
        self.hook_handle = last_attn_layer.register_forward_hook(hook)

    def extract(self, x):
        with torch.no_grad():
            if hasattr(self.model, 'encode_image'):
                _ = self.model.encode_image(x)
            elif hasattr(self.model, 'model') and hasattr(self.model.model, 'encode_image'):
                _ = self.model.model.encode_image(x)
            elif hasattr(self.model, 'get_image_features'):
                _ = self.model.get_image_features(x)
            else:
                _ = self.model(x)
        return self.features

    def remove_hook(self):
        if self.hook_handle:
            self.hook_handle.remove()

# ==========================================
# 2. LOST ALGORITHM MATH
# ==========================================
def compute_similarity_matrix(features, bias_value=0.0):
    features = F.normalize(features, p=2, dim=-1)
    gram_matrix = features @ features.T
    gram_matrix = gram_matrix + bias_value
    return torch.clamp(gram_matrix, min=-1.0, max=1.0)

def run_lost_seed_selection(gram_matrix, threshold=0.0):
    A = (gram_matrix > threshold).float()
    degrees = A.sum(dim=-1)
    seed_index = torch.argmin(degrees)
    return seed_index, gram_matrix[seed_index]

# ==========================================
# 3. CORLOC AND BOUNDING BOX MATH
# ==========================================
def extract_bounding_box(similarity_map, grid_size, orig_width, orig_height, threshold=0.0):
    """Converts the LOST map into a bounding box scaled to the original image."""
    # 1. Threshold map into binary mask
    binary_map = (similarity_map > threshold).astype(int)
    
    # 2. Find connected components (blobs)
    labeled_array, num_features = ndimage.label(binary_map)
    if num_features == 0:
        return [0, 0, orig_width, orig_height] # Fallback to whole image
        
    # 3. Find the largest blob (excluding background 0)
    sizes = np.bincount(labeled_array.ravel())
    sizes[0] = 0 
    largest_label = sizes.argmax()
    
    # 4. Get min/max grid coordinates of the largest blob
    slice_y, slice_x = ndimage.find_objects((labeled_array == largest_label).astype(int))[0]
    grid_ymin, grid_ymax = slice_y.start, slice_y.stop
    grid_xmin, grid_xmax = slice_x.start, slice_x.stop
    
    # 5. Scale back to original image dimensions
    scale_x = orig_width / grid_size
    scale_y = orig_height / grid_size
    
    xmin = int(grid_xmin * scale_x)
    ymin = int(grid_ymin * scale_y)
    xmax = int(grid_xmax * scale_x)
    ymax = int(grid_ymax * scale_y)
    
    return [xmin, ymin, xmax, ymax]

def compute_iou(boxA, boxB):
    """Calculates Intersection over Union between two boxes [xmin, ymin, xmax, ymax]"""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0: return 0.0

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    return interArea / float(boxAArea + boxBArea - interArea)

# ==========================================
# 4. DATASET EVALUATION LOOP
# ==========================================
def evaluate_dataset(config, dataset):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n--- Loading Model: {config['name']} ---")
    
    # Load Model
    if config["source"] == "timm":
        model = timm.create_model(config["model_name"], pretrained=config["pretrained"]).to(device).eval()
        num_regs = config["regs"]
    elif config["source"] == "open_clip":
        model, _, _ = open_clip.create_model_and_transforms(config["model_name"], pretrained=config["pretrained"])
        model = model.to(device).eval()
        num_regs = config["regs"]
    elif config["source"] == "hf":
        model = AutoModel.from_pretrained(config["model_name"], trust_remote_code=True).to(device).eval()
        num_regs = getattr(model, 'num_register_tokens', 4) 
        
    extractor = AttentionFeatureExtractor(model, feature_type=config["type"])
    
    # Preprocessing setup
    img_size = config["img_size"]
    grid_size = img_size // config["patch_size"]
    transform = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    correct_localizations = 0
    total_images = len(dataset)
    
    # Optional: Only run on first 500 images if you want a fast test. 
    # Change total_images to 500 for a quick debug run.
    # total_images = 500 

    print(f"Evaluating CorLoc on {total_images} images...")
    for idx in tqdm(range(total_images)):
        image, target = dataset[idx]
        orig_width, orig_height = image.size
        
        # 1. Extract Ground Truth Boxes (can be multiple per image)
        objects = target['annotation']['object']
        if not isinstance(objects, list):
            objects = [objects] # Handle single object case
            
        gt_boxes = []
        for obj in objects:
            bbox = obj['bndbox']
            gt_boxes.append([int(bbox['xmin']), int(bbox['ymin']), int(bbox['xmax']), int(bbox['ymax'])])
            
        # 2. Extract Features
        input_tensor = transform(image).unsqueeze(0).to(device)
        features = extractor.extract(input_tensor)[0]
        
        # Drop Special Tokens
        num_special_tokens = 1 + num_regs 
        patch_features = features[num_special_tokens:, :] 

        # 3. Run LOST Algorithm
        gram = compute_similarity_matrix(patch_features, bias_value=config["bias"])
        seed_idx, corr = run_lost_seed_selection(gram, threshold=0.0)
        map_result = corr.view(grid_size, grid_size).cpu().numpy()
        
        # 4. Get Predicted Box
        pred_box = extract_bounding_box(map_result, grid_size, orig_width, orig_height, threshold=0.0)
        
        # 5. Calculate Max IoU against all ground truth boxes
        max_iou = 0.0
        for gt_box in gt_boxes:
            iou = compute_iou(pred_box, gt_box)
            if iou > max_iou:
                max_iou = iou
                
        # 6. CorLoc Threshold
        if max_iou >= 0.5:
            correct_localizations += 1

    extractor.remove_hook()
    
    corloc_score = (correct_localizations / total_images) * 100
    print(f"Result for {config['name']}: CorLoc = {corloc_score:.2f}%")
    return corloc_score

if __name__ == "__main__":
    # Download and load PASCAL VOC 2007 (Train/Val split is standard for this task)
    print("Preparing PASCAL VOC 2007 Dataset...")
    # NOTE: Set download=True the first time you run this! It will download ~450MB.
    voc_dataset = VOCDetection(root="./src/Raffo/data", year="2007", image_set="trainval", download=True)
    
    configs = [
        {"name": "DINOv2_NoReg",   "source": "timm", "model_name": "vit_base_patch14_dinov2.lvd142m", "pretrained": True, "type": "keys",   "bias": 0.0, "img_size": 518, "patch_size": 14, "regs": 0},
        {"name": "DINOv2_WithReg", "source": "timm", "model_name": "vit_base_patch14_reg4_dinov2.lvd142m", "pretrained": True, "type": "keys",   "bias": 0.0, "img_size": 518, "patch_size": 14, "regs": 4},
        
        {"name": "OpenCLIP_NoReg", "source": "open_clip", "model_name": "ViT-B-16", "pretrained": "laion2b_s34b_b88k", "type": "values", "bias": 0.1, "img_size": 224, "patch_size": 16, "regs": 0},
        {"name": "OpenCLIP_TestTimeReg", "source": "hf", "model_name": "amildravid4292/clip-vitb16-test-time-registers", "pretrained": True, "type": "values", "bias": 0.1, "img_size": 224, "patch_size": 16, "regs": "dynamic"},
    ]

    final_results = {}
    for cfg in configs:
        score = evaluate_dataset(cfg, voc_dataset)
        final_results[cfg["name"]] = score
        
    print("\n==============================")
    print("FINAL TABLE 3 RESULTS (VOC07)")
    print("==============================")
    for name, score in final_results.items():
        print(f"{name:<25}: {score:.1f}%")