import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerConfig, SegformerForSemanticSegmentation

class SegFormerWrapper(nn.Module):
    """
    Wrapper around HuggingFace's SegFormer for semantic segmentation.
    This model utilizes Hierarchical Vision Transformers (ViT) in the encoder
    and a lightweight MLP decoder.
    
    Ref: https://arxiv.org/abs/2105.15203
    """
    def __init__(
        self,
        pretrained_model_name: str = "nvidia/mit-b0",
        num_classes: int = 1,
        from_pretrained: bool = False
    ):
        super().__init__()
        self.num_classes = num_classes
        
        if from_pretrained:
            self.model = SegformerForSemanticSegmentation.from_pretrained(
                pretrained_model_name,
                num_labels=num_classes,
                ignore_mismatched_sizes=True
            )
        else:
            config = SegformerConfig.from_pretrained(pretrained_model_name)
            config.num_labels = num_classes
            self.model = SegformerForSemanticSegmentation(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.model(pixel_values=x)
        logits = outputs.logits  # Shape: [B, num_classes, H/4, W/4]
        
        scaled_logits = F.interpolate(
            logits,
            size=x.shape[2:],  # Resize back to (H, W)
            mode="bilinear",
            align_corners=False
        )
        return scaled_logits
