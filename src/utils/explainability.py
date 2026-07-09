import torch
import torch.nn as nn
import numpy as np
import cv2
from typing import Tuple, Optional

class SegmentationGradCAM:
    """
    Computes Grad-CAM (Gradient-weighted Class Activation Mapping) for segmentation networks.
    Hooks into a target convolutional layer to extract features and gradients.
    """
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None
        self.handlers = []

        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.handlers.append(target_layer.register_forward_hook(forward_hook))
        self.handlers.append(target_layer.register_full_backward_hook(backward_hook))

    def remove_hooks(self):
        for handler in self.handlers:
            handler.remove()

    def generate_cam(self, input_tensor: torch.Tensor, class_idx: int = 0) -> np.ndarray:
        self.model.zero_grad()
        output = self.model(input_tensor)
        
        # Target score is the average/sum of positive outputs
        loss = output[0, class_idx].sum()
        loss.backward()

        if self.gradients is None or self.activations is None:
            h, w = input_tensor.shape[2:]
            return np.zeros((h, w), dtype=np.float32)

        gradients = self.gradients[0]
        activations = self.activations[0]

        weights = torch.mean(gradients, dim=(1, 2), keepdim=True)
        cam = torch.sum(weights * activations, dim=0).cpu().numpy()

        cam = np.maximum(cam, 0)
        
        h, w = input_tensor.shape[2:]
        cam = cv2.resize(cam, (w, h))
        
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-7:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)
            
        return cam

def overlay_heatmap(image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    else:
        image = image.astype(np.uint8)

    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    color_heatmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    color_heatmap = cv2.cvtColor(color_heatmap, cv2.COLOR_BGR2RGB)
    
    blended = cv2.addWeighted(image, 1.0 - alpha, color_heatmap, alpha, 0)
    return blended
