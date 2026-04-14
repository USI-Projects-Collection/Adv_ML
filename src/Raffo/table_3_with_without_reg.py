import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import timm
import open_clip 
from transformers import AutoModel

# ==========================================
# 1. DYNAMIC ATTENTION EXTRACTOR
# ==========================================
# ==========================================
# 1. DYNAMIC ATTENTION EXTRACTOR
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
        # 1. Identify which library the model comes from
        if hasattr(self.model, 'blocks'): 
            # Standard timm model (DINOv2)
            last_attn_layer = self.model.blocks[-1].attn
            self.model_backend = "timm"
            
        elif hasattr(self.model, 'visual') and hasattr(self.model.visual, 'transformer'): 
            # Standard open_clip model
            last_attn_layer = self.model.visual.transformer.resblocks[-1].attn
            self.model_backend = "open_clip"
            
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'visual') and hasattr(self.model.model.visual, 'transformer'):
            # Wrapped OpenCLIP (The Hugging Face Test-Time Registers model)
            last_attn_layer = self.model.model.visual.transformer.resblocks[-1].attn
            self.model_backend = "open_clip"
            
        elif hasattr(self.model, 'vision_model'): 
            # Standard HuggingFace CLIP fallback
            last_attn_layer = self.model.vision_model.encoder.layers[-1].self_attn
            self.model_backend = "hf_clip"
            
        else:
            raise NotImplementedError("Could not identify the model architecture.")
        
        # 2. Define the interception hook
        def hook(module, input, output):
            x = input[0]
            B, N, C = x.shape
            
            if self.model_backend == "timm":
                qkv = module.qkv(x)
                qkv = qkv.reshape(B, N, 3, module.num_heads, C // module.num_heads).permute(2, 0, 3, 1, 4)
                feat = qkv[1] if self.feature_type == "keys" else qkv[2]
                self.features = feat.transpose(1, 2).reshape(B, N, C)
                
            elif self.model_backend == "open_clip":
                # OpenCLIP uses separate weights for the projection
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
            # Trigger the forward pass depending on where the encode function is hiding
            if hasattr(self.model, 'encode_image'):
                _ = self.model.encode_image(x)
            elif hasattr(self.model, 'model') and hasattr(self.model.model, 'encode_image'):
                _ = self.model.model.encode_image(x) # Wrapped OpenCLIP
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
# 3. EVALUATION FUNCTION
# ==========================================
def evaluate_model_for_lost(image_path, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Preprocessing
    img_size = config["img_size"]
    grid_size = img_size // config["patch_size"]
    image = Image.open(image_path).convert("RGB")
    
    transform = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = transform(image).unsqueeze(0).to(device)

    print(f"\nLoading {config['name']}...")
    
    # 2. Load the specific model architecture
    if config["source"] == "timm":
        model = timm.create_model(config["model_name"], pretrained=config["pretrained"]).to(device).eval()
        num_regs = config["regs"]
        
    elif config["source"] == "open_clip":
        model, _, _ = open_clip.create_model_and_transforms(config["model_name"], pretrained=config["pretrained"])
        model = model.to(device).eval()
        num_regs = config["regs"]
        
    elif config["source"] == "hf":
        # Option 2: The Test-Time Registers Model!
        # trust_remote_code=True is required because the authors wrote custom forward pass logic
        model = AutoModel.from_pretrained(config["model_name"], trust_remote_code=True).to(device).eval()
        # The test-time model has an attribute for its dynamically added registers
        num_regs = getattr(model, 'num_register_tokens', 4) 
    
    # 3. Extract Features
    extractor = AttentionFeatureExtractor(model, feature_type=config["type"])
    features = extractor.extract(input_tensor)[0]
    extractor.remove_hook()

    # 4. Drop Special Tokens ([CLS] + dynamic [REG]s)
    num_special_tokens = 1 + num_regs 
    patch_features = features[num_special_tokens:, :] 

    # 5. Run LOST
    gram = compute_similarity_matrix(patch_features, bias_value=config["bias"])
    seed_idx, corr = run_lost_seed_selection(gram, threshold=0.0)
    
    map_result = corr.view(grid_size, grid_size).cpu().numpy()
    return map_result, image.resize((img_size, img_size))

# ==========================================
# 4. RUNNER
# ==========================================
def run_table_3_experiment():
    img_path = "./src/Raffo/img/Black_Labrador_Retriever_portrait.jpg"
    
    # The dictionary containing the specific rules for each model
    configs = [
        # --- DINOv2 Models (Trained) ---
        {"name": "DINOv2_NoReg",   "source": "timm", "model_name": "vit_base_patch14_dinov2.lvd142m", "pretrained": True, "type": "keys",   "bias": 0.0, "img_size": 518, "patch_size": 14, "regs": 0},
        {"name": "DINOv2_WithReg", "source": "timm", "model_name": "vit_base_patch14_reg4_dinov2.lvd142m", "pretrained": True, "type": "keys",   "bias": 0.0, "img_size": 518, "patch_size": 14, "regs": 4},
        
        # --- OpenCLIP Models ---
        {"name": "OpenCLIP_NoReg", "source": "open_clip", "model_name": "ViT-B-16", "pretrained": "laion2b_s34b_b88k", "type": "values", "bias": 0.1, "img_size": 224, "patch_size": 16, "regs": 0},
        # Option 2: The Test-Time Registers variant directly from HuggingFace
        {"name": "OpenCLIP_TestTimeReg", "source": "hf", "model_name": "amildravid4292/clip-vitb16-test-time-registers", "pretrained": True, "type": "values", "bias": 0.1, "img_size": 224, "patch_size": 16, "regs": "dynamic"},
    ]

    results = {}
    for cfg in configs:
        map_result, img = evaluate_model_for_lost(img_path, cfg)
        results[cfg["name"]] = (map_result, img)

    # Visualization
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    for i, cfg in enumerate(configs):
        name = cfg["name"]
        map_result, img = results[name]
        
        # Plot Original Image
        axes[i].set_title(f"{name}\nOriginal Image")
        axes[i].imshow(img)
        axes[i].axis('off')
        
        # Plot LOST Map
        axes[i+4].set_title(f"LOST Map")
        axes[i+4].imshow(map_result, cmap='viridis')
        axes[i+4].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_table_3_experiment()