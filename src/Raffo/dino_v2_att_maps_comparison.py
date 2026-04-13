import torch
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import timm

# ==========================================
# 1. THE PYTORCH HOOK (Extracting Attention Maps)
# ==========================================
class AttentionMapExtractor:
    """
    Attaches to the last self-attention block to compute the raw attention probabilities.
    """
    def __init__(self, model):
        self.model = model
        self.attention_maps = None
        self.hook_handle = None
        self._register_hook()

    def _register_hook(self):
        last_attn_layer = self.model.blocks[-1].attn
        scale = last_attn_layer.scale # The 1 / sqrt(d) scaling factor
        
        def hook(module, input, output):
            x = input[0]
            B, N, C = x.shape
            
            # Pass through the linear layer and separate Q and K
            qkv = module.qkv(x)
            qkv = qkv.reshape(B, N, 3, module.num_heads, C // module.num_heads)
            qkv = qkv.permute(2, 0, 3, 1, 4) 
            q, k = qkv[0], qkv[1]
            
            # Compute the attention probabilities matrix
            attn = (q @ k.transpose(-2, -1)) * scale
            attn = attn.softmax(dim=-1)
            
            # Save the attention maps: shape [Batch, num_heads, Seq_len, Seq_len]
            self.attention_maps = attn

        self.hook_handle = last_attn_layer.register_forward_hook(hook)

    def extract(self, x):
        with torch.no_grad():
            _ = self.model(x)
        return self.attention_maps

    def remove_hook(self):
        if self.hook_handle:
            self.hook_handle.remove()

# ==========================================
# 2. MAIN PIPELINE
# ==========================================
def run_attention_experiment():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Local Image (Using the updated 518x518 resolution)
    url = "src/Raffo/img/Black_Labrador_Retriever_portrait.jpg"
    image = Image.open(url).convert("RGB")
    
    grid_size = 37 # 518 / 14 = 37x37 patches
    transform = T.Compose([
        T.Resize((518, 518)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = transform(image).unsqueeze(0).to(device)

    # 2. Load DINOv2 with 4 registers
    print("Loading DINOv2 WITH 4 registers...")
    model_reg = timm.create_model('vit_base_patch14_reg4_dinov2.lvd142m', pretrained=True).to(device).eval()

    # 3. Setup Extractor
    extractor = AttentionMapExtractor(model_reg)

    # 4. Extract Attention Maps
    print("Extracting Attention Maps...")
    attn_maps = extractor.extract(input_tensor) # Shape: [1, num_heads, N, N]
    extractor.remove_hook()

    # 5. Average across all 12 attention heads to get the global attention pattern
    # Shape becomes: [N, N] where N is 1374 (1 CLS + 4 REG + 1369 Patches)
    attn_avg = attn_maps[0].mean(dim=0).cpu().numpy()

    # 6. Slice the matrix to see what the CLS and REG tokens are paying attention to.
    # We slice [5:] on the columns to only look at attention directed towards the image patches.
    cls_attn  = attn_avg[0, 5:].reshape(grid_size, grid_size)
    reg0_attn = attn_avg[1, 5:].reshape(grid_size, grid_size)
    reg1_attn = attn_avg[2, 5:].reshape(grid_size, grid_size)
    reg2_attn = attn_avg[3, 5:].reshape(grid_size, grid_size)
    reg3_attn = attn_avg[4, 5:].reshape(grid_size, grid_size)

    # 7. Visualize Results
    fig, axes = plt.subplots(1, 6, figsize=(20, 4))
    
    axes[0].set_title("Original Image")
    axes[0].imshow(image.resize((518, 518)))
    axes[0].axis('off')

    maps = [cls_attn, reg0_attn, reg1_attn, reg2_attn, reg3_attn]
    titles = ["[CLS]", "[reg0]", "[reg1]", "[reg2]", "[reg3]"]

    for i, (m, t) in enumerate(zip(maps, titles)):
        axes[i+1].set_title(f"{t} Attention")
        # 'magma' or 'inferno' perfectly matches the aesthetic of Figure 9
        axes[i+1].imshow(m, cmap='magma') 
        axes[i+1].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_attention_experiment()