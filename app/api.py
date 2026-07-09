import os
import io
import torch
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import List, Tuple, Optional
import json

# Package imports
from src.config import load_config
from src.models.unet import UNet
from src.models.segformer import SegFormerWrapper
from src.models.sam_helper import SAMHelper
from src.utils.explainability import SegmentationGradCAM, overlay_heatmap
from src.utils.onnx_export import run_benchmark

app = FastAPI(
    title="VisionGuard AI - Segmentation API",
    description="REST API for comparing U-Net, SegFormer, and interactive SAM predictions with Grad-CAM explanations.",
    version="1.0.0"
)

# Load configuration
config = load_config()
device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

# Global model cache to avoid reloading weights on every request
MODELS = {
    "unet": None,
    "segformer": None,
    "sam": None
}

def get_model(model_type: str):
    """Dynamically loads and caches models on demand."""
    if MODELS[model_type] is not None:
        return MODELS[model_type]
        
    if model_type == "unet":
        model = UNet(
            in_channels=config.unet_in_channels,
            num_classes=config.unet_num_classes,
            bilinear=config.unet_bilinear
        )
        checkpoint_path = os.path.join(config.checkpoint_dir, "checkpoint_epoch5.pth")
        if os.path.exists(checkpoint_path):
            try:
                checkpoint = torch.load(checkpoint_path, map_location=device)
                state = checkpoint.get("model_state_dict", checkpoint)
                model.load_state_dict(state, strict=False)
                print("Loaded U-Net weights from checkpoint.")
            except Exception as e:
                print(f"Failed to load U-Net checkpoint: {e}. Using initialized weights.")
        model.to(device)
        model.eval()
        MODELS["unet"] = model
        
    elif model_type == "segformer":
        model = SegFormerWrapper(
            pretrained_model_name=config.segformer_name,
            num_classes=config.segformer_num_classes,
            from_pretrained=True
        )
        checkpoint_path = os.path.join(config.checkpoint_dir, "checkpoint_segformer.pth")
        if os.path.exists(checkpoint_path):
            try:
                checkpoint = torch.load(checkpoint_path, map_location=device)
                state = checkpoint.get("model_state_dict", checkpoint)
                model.load_state_dict(state, strict=False)
                print("Loaded SegFormer weights from checkpoint.")
            except Exception as e:
                print(f"Failed to load SegFormer checkpoint: {e}. Using pretrained weights.")
        model.to(device)
        model.eval()
        MODELS["segformer"] = model
        
    elif model_type == "sam":
        sam_helper = SAMHelper(model_name="facebook/sam-vit-base")
        sam_helper.initialize()
        MODELS["sam"] = sam_helper
        
    return MODELS[model_type]

# Preprocessing transform matching training/validation
preprocess_transform = A.Compose([
    A.Resize(height=config.image_size, width=config.image_size),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

def preprocess_image(image: Image.Image) -> Tuple[torch.Tensor, np.ndarray]:
    """Preprocesses a PIL image and returns the normalized tensor and RGB numpy array."""
    img_rgb = np.array(image.convert("RGB"))
    transformed = preprocess_transform(image=img_rgb)
    tensor = transformed["image"].unsqueeze(0).to(device)  # Add batch size [1, C, H, W]
    return tensor, img_rgb

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "device": str(device),
        "unet_cached": MODELS["unet"] is not None,
        "segformer_cached": MODELS["segformer"] is not None,
        "sam_cached": MODELS["sam"] is not None
    }

@app.post("/predict")
async def predict_segmentation(
    file: UploadFile = File(...),
    model_type: str = Form("unet")
):
    if model_type not in ["unet", "segformer"]:
        raise HTTPException(status_code=400, detail="Invalid model type. Choose 'unet' or 'segformer'")
        
    # Read and parse uploaded image
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file uploaded.")
        
    # Process
    tensor, _ = preprocess_image(image)
    model = get_model(model_type)
    
    with torch.no_grad():
        logits = model(tensor)
        
    num_classes = getattr(model, 'n_classes', getattr(model, 'num_classes', 1))
    if num_classes == 1:
        probs = torch.sigmoid(logits).squeeze(0).squeeze(0)  # Shape: [H, W]
        mask = (probs > 0.5).cpu().numpy().astype(np.uint8) * 255
    else:
        probs = torch.softmax(logits, dim=1).squeeze(0)      # Shape: [C, H, W]
        mask = probs.argmax(dim=0).cpu().numpy().astype(np.uint8) * (255 // (num_classes - 1))
        
    # Resize mask back to original image size
    mask_pil = Image.fromarray(mask).resize(image.size, resample=Image.Resampling.NEAREST)
    
    # Save back to a binary stream
    img_io = io.BytesIO()
    mask_pil.save(img_io, 'PNG')
    img_io.seek(0)
    
    return StreamingResponse(img_io, media_type="image/png")

@app.post("/sam/predict")
async def predict_sam_mask(
    file: UploadFile = File(...),
    points: str = Form("[[100, 100]]")
):
    try:
        points_list = json.loads(points)
        if not isinstance(points_list, list) or len(points_list) == 0:
            raise ValueError()
    except Exception:
        raise HTTPException(status_code=400, detail="Points parameter must be a JSON array of coordinate pairs: e.g. '[[100, 150]]'")
        
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file uploaded.")
        
    sam_helper = get_model("sam")
    # points_list formatted as list of (x, y) coordinates
    input_points = [(pt[0], pt[1]) for pt in points_list]
    
    mask = sam_helper.predict_mask(image, input_points)
    mask_image = Image.fromarray(mask * 255).convert("L")
    
    img_io = io.BytesIO()
    mask_image.save(img_io, 'PNG')
    img_io.seek(0)
    
    return StreamingResponse(img_io, media_type="image/png")

@app.post("/explain")
async def explain_segmentation(
    file: UploadFile = File(...),
    model_type: str = Form("unet"),
    alpha: float = Form(0.4)
):
    if model_type not in ["unet", "segformer"]:
        raise HTTPException(status_code=400, detail="Invalid model type. Choose 'unet' or 'segformer'")
        
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file uploaded.")
        
    tensor, original_rgb = preprocess_image(image)
    model = get_model(model_type)
    
    # Identify target hook layer
    if model_type == "unet":
        # Hook on last double conv in decoder
        target_layer = model.up4.conv.double_conv[3]
    else:
        # Segformer target layer (the last normalization layer in the final encoder stage)
        target_layer = model.model.segformer.encoder.patch_embeddings[3].layer_norm
        
    grad_cam = SegmentationGradCAM(model, target_layer)
    
    try:
        # Enable gradient calculation to allow backpropagation
        with torch.enable_grad():
            tensor.requires_grad = True
            heatmap = grad_cam.generate_cam(tensor, class_idx=0)
            
        grad_cam.remove_hooks()
        
        # Resize original image to match tensor size (where Grad-CAM was computed)
        resized_orig = cv2.resize(original_rgb, (config.image_size, config.image_size))
        blended = overlay_heatmap(resized_orig, heatmap, alpha=alpha)
        
        # Scale back to original image resolution for presentation
        blended_pil = Image.fromarray(blended).resize(image.size, resample=Image.Resampling.BILINEAR)
        
        img_io = io.BytesIO()
        blended_pil.save(img_io, 'PNG')
        img_io.seek(0)
        
        return StreamingResponse(img_io, media_type="image/png")
    except Exception as e:
        grad_cam.remove_hooks()
        raise HTTPException(status_code=500, detail=f"Grad-CAM generation failed: {e}")

@app.get("/benchmark")
def get_benchmark_report():
    # If ONNX file exists, run a live benchmark.
    onnx_path = config.onnx_export_path
    if not os.path.exists(onnx_path):
        # Trigger an export to verify
        from src.utils.onnx_export import export_unet_to_onnx
        try:
            export_unet_to_onnx()
        except Exception as e:
            return {
                "error": f"ONNX export failed, cannot benchmark: {e}"
            }
            
    # Load model and run
    model = get_model("unet")
    dummy_input = torch.randn(1, config.unet_in_channels, config.image_size, config.image_size)
    report = run_benchmark(model, onnx_path, dummy_input)
    return report

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.api:app", host=config.api_host, port=config.api_port, reload=True)
