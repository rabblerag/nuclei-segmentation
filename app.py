import os
import gradio as gr
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
import numpy as np
import cv2
from skimage.measure import label, regionprops
import pandas as pd
import matplotlib.pyplot as plt

# --- 1. Architecture Definition ---
class ResUNet(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        
        self.enc1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu) 
        self.maxpool = resnet.maxpool
        self.enc2 = resnet.layer1
        self.enc3 = resnet.layer2
        self.enc4 = resnet.layer3
        self.bottleneck = resnet.layer4
        
        self.up4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec4 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.up1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.up0 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.out_conv = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)                
        e2 = self.enc2(self.maxpool(e1)) 
        e3 = self.enc3(e2)               
        e4 = self.enc4(e3)               
        b = self.bottleneck(e4)          
        
        d4 = self.up4(b)                 
        d4 = torch.cat([d4, e4], dim=1)  
        d4 = self.dec4(d4)               
        
        d3 = self.up3(d4)                
        d3 = torch.cat([d3, e3], dim=1)  
        d3 = self.dec3(d3)
        
        d2 = self.up2(d3)                
        d2 = torch.cat([d2, e2], dim=1)  
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)                
        d1 = torch.cat([d1, e1], dim=1)  
        d1 = self.dec1(d1)
        
        d0 = self.up0(d1)                
        out = self.out_conv(d0)          
        return out

# --- 2. Setup Model ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ResUNet().to(device)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
weights_path = os.path.join(BASE_DIR, 'outputs', 'model_weights', 'resunet_best.pth')
try:
    model.load_state_dict(torch.load(weights_path, map_location=device))
    print(f"Loaded weights from {weights_path}")
except Exception as e:
    print(f"Warning: Could not load weights from {weights_path}. Error: {e}")

model.eval()

# --- 3. Processing Functions ---
def generate_bio_art(mask_np):
    num_labels, labels = cv2.connectedComponents(mask_np.astype(np.uint8))
    
    np.random.seed(42)
    colors = np.random.randint(50, 255, size=(num_labels, 3), dtype=np.uint8)
    colors[0] = [0, 0, 0] # Background is black
    
    art_img = colors[labels]
    return Image.fromarray(art_img)

def analyze_cells(mask_np):
    labeled_mask = label(mask_np > 0)
    props = regionprops(labeled_mask)
    
    data = []
    for i, p in enumerate(props):
        data.append({
            "Cell ID": i + 1,
            "Area (pixels)": p.area,
            "Perimeter (pixels)": round(p.perimeter, 2),
            "Eccentricity": round(p.eccentricity, 3),
            "Solidity": round(p.solidity, 3),
            "Major Axis (px)": round(p.axis_major_length, 2),
            "Minor Axis (px)": round(p.axis_minor_length, 2),
            "Equivalent Diameter": round(p.equivalent_diameter_area, 2),
            "Orientation (rad)": round(p.orientation, 3)
        })
    df = pd.DataFrame(data)
    
    fig, ax = plt.subplots(figsize=(6, 4))
    if not df.empty:
        sc = ax.scatter(df["Area (pixels)"], df["Perimeter (pixels)"], 
                        c=df["Eccentricity"], cmap='plasma', alpha=0.8, edgecolors='w')
        plt.colorbar(sc, label="Eccentricity (0=Circle, 1=Line)")
    ax.set_title("Cell Health: Area vs Perimeter")
    ax.set_xlabel("Area")
    ax.set_ylabel("Perimeter")
    ax.grid(True, linestyle='--', alpha=0.6)
    
    fig.canvas.draw()
    plot_img = np.array(fig.canvas.buffer_rgba())
    plt.close(fig)
    return df, Image.fromarray(plot_img)

def process_image(image):
    if image is None:
        return None, None, None, None, None
    
    orig_size = image.size
    img_resized = image.resize((256, 256), Image.BILINEAR).convert('RGB')
    tensor_img = T.ToTensor()(img_resized).unsqueeze(0).to(device)
    
    with torch.no_grad():
        pred = model(tensor_img)
        pred_sig = torch.sigmoid(pred).squeeze().cpu().numpy()
        
    mask_binary = (pred_sig > 0.5).astype(np.uint8) * 255
    mask_img = Image.fromarray(mask_binary).resize(orig_size, Image.NEAREST)
    mask_np = np.array(mask_img)
    
    overlay = np.array(image.convert('RGB'))
    overlay[mask_np == 255] = overlay[mask_np == 255] * 0.5 + np.array([255, 0, 0]) * 0.5
    overlay_img = Image.fromarray(overlay.astype(np.uint8))
    
    art_img = generate_bio_art(mask_np)
    df, plot_img = analyze_cells(mask_np)
    
    return mask_img, overlay_img, art_img, df, plot_img

# --- 4. Gradio Interface ---
with gr.Blocks(title="Nuclei Segmentation & Analysis") as demo:
    gr.Markdown("# 🔬 Interactive Cell Scanner & Analyzer")
    gr.Markdown("Upload an image to segment nuclei, generate abstract Bio-Art, and analyze cell health metrics!")
    
    with gr.Row():
        with gr.Column():
            img_in = gr.Image(type="pil", label="Input Image")
            btn = gr.Button("Analyze Image", variant="primary")
        
        with gr.Column():
            with gr.Tabs():
                with gr.TabItem("Segmentation"):
                    mask_out = gr.Image(type="pil", label="Predicted Mask")
                    overlay_out = gr.Image(type="pil", label="Overlay")
                with gr.TabItem("Bio-Art"):
                    art_out = gr.Image(type="pil", label="Abstract Bio-Art")
                with gr.TabItem("Cell Health Analysis"):
                    plot_out = gr.Image(type="pil", label="Area vs Perimeter Scatter Plot")
                    df_out = gr.Dataframe(label="Cell Metrics")
                    
    btn.click(fn=process_image, inputs=img_in, outputs=[mask_out, overlay_out, art_out, df_out, plot_out])

if __name__ == "__main__":
    demo.launch(share=False)
