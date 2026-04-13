import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
import requests
import matplotlib.pyplot as plt
import timm

# ==========================================
# 1. THE PYTORCH HOOK (Extracting KEYS for DINOv2)
# ==========================================
class AttentionKeyExtractor:
    def __init__(self, model):
        self.model = model
        self.keys = None
        self.hook_handle = None
        self._register_hook()

    def _register_hook(self):
        # Attach to the last attention block
        last_attn_layer = self.model.blocks[-1].attn
        
        def hook(module, input, output):
            x = input[0]
            B, N, C = x.shape
            
            qkv = module.qkv(x)
            qkv = qkv.reshape(B, N, 3, module.num_heads, C // module.num_heads)
            qkv = qkv.permute(2, 0, 3, 1, 4) 
            
            # --- CRITICAL CHANGE FOR DINOv2 ---
            # Index 1 is KEYS. (DeiT/OpenCLIP used Values at Index 2)
            k = qkv[1] 
            
            self.keys = k.transpose(1, 2).reshape(B, N, C)

        self.hook_handle = last_attn_layer.register_forward_hook(hook)

    def extract(self, x):
        with torch.no_grad():
            _ = self.model(x)
        return self.keys

    def remove_hook(self):
        if self.hook_handle:
            self.hook_handle.remove()


# ==========================================
# 1. THE FUNCTIONS (From previous step)
# ==========================================
def compute_similarity_matrix(features, bias_value=0.0):
    # L2 normalize the features to get cosine similarity to focus on direction rather than magnitude (semantic similarity)
    features = F.normalize(features, p=2, dim=-1)
    # Create a NxN Gram matrix where [i,j] is the cosine similarity between patch i and patch j
    gram_matrix = features @ features.T
    gram_matrix = gram_matrix + bias_value
    return torch.clamp(gram_matrix, min=-1.0, max=1.0)

def run_lost_seed_selection(gram_matrix, threshold=0.0):
    num_patches = gram_matrix.shape[0]
    # Turn the Gram matrix into a binary adjacency matrix. If the similarity is above the threshold, we consider those patches connected.
    A = (gram_matrix > threshold).float()
    # Calculate the degree of each patch (how many other patches it is similar to)
    degrees = A.sum(dim=-1)
    
    # The patch with the lowest degree is distinct from the background
    seed_index = torch.argmin(degrees)
    seed_correlations = gram_matrix[seed_index]
    
    return seed_index, seed_correlations

# ==========================================
# 2. SIMULATE DATA (Mocking the ViT output)
# ==========================================
def create_mock_features():
    """
    Simulates a 14x14 grid of patches (196 total patches).
    We will make the 'background' uniform and put an 'object' in the center.
    """
    grid_size = 14
    embed_dim = 384 # Standard for small ViTs
    
    # 1. Create a uniform background
    # All background patches get the same base vector with a tiny bit of noise
    base_bg = torch.ones(embed_dim)
    features = base_bg.unsqueeze(0).repeat(grid_size * grid_size, 1) 
    features += torch.randn_like(features) * 0.1 
    
    # 2. Inject an "object" in the middle (e.g., a 4x4 patch area)
    base_obj = torch.zeros(embed_dim)
    base_obj[0:50] = 5.0 # Make it distinct from the background
    
    # Map 1D index to 2D grid and place the object
    for y in range(5, 9):
        for x in range(5, 9):
            idx = y * grid_size + x
            features[idx] = base_obj + torch.randn(embed_dim) * 0.1
            
    return features, grid_size

# ==========================================
# 3. RUN THE TEST AND VISUALIZE
# ==========================================
def test_lost_pipeline():
    print("Generating mock feature map (14x14 grid)...")
    patch_features, grid_size = create_mock_features()
    
    print("Computing Gram Matrix...")
    # Testing the bias tweak mentioned in the paper
    gram_matrix = compute_similarity_matrix(patch_features, bias_value=0.1)
    
    print("Running LOST seed selection...")
    # threshold > 0 usually works best after the dot product
    seed_index, seed_correlations = run_lost_seed_selection(gram_matrix, threshold=0.5)
    
    # Calculate grid coordinates of the seed to verify
    seed_y = (seed_index.item() // grid_size)
    seed_x = (seed_index.item() % grid_size)
    print(f"Seed found at patch index: {seed_index.item()} (Grid coordinates: y={seed_y}, x={seed_x})")
    
    # Reshape the correlations back into a 2D image format to visualize
    correlation_map = seed_correlations.view(grid_size, grid_size).numpy()
    
    # Plotting
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.title("Step 1: The 'Object' Seed Selection")
    plt.imshow(correlation_map, cmap='viridis')
    plt.plot(seed_x, seed_y, 'r*', markersize=15, label='Selected Seed')
    plt.legend()
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.title("Step 2: Seed Expansion (Similarity Map)")
    # We threshold the map just like LOST does to find the whole object
    plt.imshow(correlation_map > 0.5, cmap='gray') 
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# ==========================================
# 3. MAIN PIPELINE
# ==========================================
def run_experiment():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load a sample image (e.g., a dog)
    url = "src/Raffo/img/Black_Labrador_Retriever_portrait.jpg"
    # image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
    image = Image.open(url).convert("RGB")
    
    # DINOv2 uses patch size 14. 518 / 14 = 37x37 grid
    grid_size = 37 
    transform = T.Compose([
        T.Resize((518, 518)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = transform(image).unsqueeze(0).to(device)

    # 2. Load the Models
    print("Loading DINOv2 without registers...")
    model_no_reg = timm.create_model('vit_base_patch14_dinov2.lvd142m', pretrained=True).to(device).eval()
    
    print("Loading DINOv2 WITH registers...")
    model_reg_4 = timm.create_model('vit_base_patch14_reg4_dinov2.lvd142m', pretrained=True).to(device).eval()

    # 3. Setup Extractors
    extractor_no_reg = AttentionKeyExtractor(model_no_reg)
    extractor_reg_4 = AttentionKeyExtractor(model_reg_4)

    # 4. Extract Features (Keys)
    print("Extracting Keys...")
    keys_no_reg = extractor_no_reg.extract(input_tensor)[0]
    keys_reg_4 = extractor_reg_4.extract(input_tensor)[0]       

    extractor_no_reg.remove_hook()
    extractor_reg_4.remove_hook()

    # 5. Drop Special Tokens
    # DINOv2 baseline has 1 CLS token
    patch_features_no_reg = keys_no_reg[1:, :] 
    
    # DINOv2 with registers has 1 CLS token + 4 REG tokens = 5 special tokens
    patch_features_reg_4 = keys_reg_4[5:, :] 

    # 6. Run LOST Algorithm
    print("Running LOST...")
    gram_no_reg = compute_similarity_matrix(patch_features_no_reg, bias_value=0.0)
    seed_idx_no_reg, corr_no_reg = run_lost_seed_selection(gram_no_reg, threshold=0.0)

    gram_reg_4 = compute_similarity_matrix(patch_features_reg_4, bias_value=0.0)
    seed_idx_reg_4, corr_reg_4 = run_lost_seed_selection(gram_reg_4, threshold=0.0)

    map_no_reg = corr_no_reg.view(grid_size, grid_size).cpu().numpy()
    map_reg_4 = corr_reg_4.view(grid_size, grid_size).cpu().numpy()

    # 7. Visualize Results
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].set_title("Original Image")
    axes[0].imshow(image.resize((224, 224)))
    axes[0].axis('off')

    axes[1].set_title("DINOv2 (No Registers)\nLOST Similarity Map")
    axes[1].imshow(map_no_reg, cmap='viridis')
    axes[1].axis('off')

    axes[2].set_title("DINOv2 (WITH 4 Registers)\nLOST Similarity Map")
    axes[2].imshow(map_reg_4, cmap='viridis')
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # test_lost_pipeline()
    run_experiment()