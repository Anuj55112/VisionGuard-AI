import os
import yaml
from pathlib import Path
from typing import Any, Dict

class Config:
    def __init__(self, config_dict: Dict[str, Any]):
        self.raw = config_dict
        
        # Model config
        model = config_dict.get("model", {})
        self.unet_in_channels: int = model.get("unet", {}).get("in_channels", 3)
        self.unet_num_classes: int = model.get("unet", {}).get("num_classes", 1)
        self.unet_bilinear: bool = model.get("unet", {}).get("bilinear", True)
        
        self.segformer_name: str = model.get("segformer", {}).get("pretrained_model_name", "nvidia/mit-b0")
        self.segformer_num_classes: int = model.get("segformer", {}).get("num_classes", 1)
        
        self.sam_checkpoint_url: str = model.get("sam", {}).get("checkpoint_url", "")
        self.sam_model_type: str = model.get("sam", {}).get("model_type", "vit_b")
        
        # Dataset config
        dataset = config_dict.get("dataset", {})
        self.image_size: int = dataset.get("image_size", 256)
        self.batch_size: int = dataset.get("batch_size", 8)
        self.val_split: float = dataset.get("val_split", 0.2)
        self.num_workers: int = dataset.get("num_workers", 0)
        
        # Training config
        training = config_dict.get("training", {})
        self.epochs: int = training.get("epochs", 5)
        self.learning_rate: float = training.get("learning_rate", 0.001)
        self.weight_decay: float = training.get("weight_decay", 0.0001)
        self.mixed_precision: bool = training.get("mixed_precision", True)
        self.checkpoint_dir: str = training.get("checkpoint_dir", "checkpoints")
        self.wandb_logging: bool = training.get("wandb_logging", False)
        self.wandb_project: str = training.get("wandb_project", "visionguard-ai")
        
        # ONNX config
        onnx = config_dict.get("onnx", {})
        self.onnx_export_path: str = onnx.get("export_path", "checkpoints/unet_model.onnx")
        self.onnx_opset_version: int = onnx.get("opset_version", 15)
        
        # API config
        api = config_dict.get("api", {})
        self.api_host: str = api.get("host", "0.0.0.0")
        self.api_port: int = api.get("port", 8000)

def load_config(config_path: str = "") -> Config:
    if not config_path:
        current_dir = Path(__file__).resolve().parent
        config_path = str(current_dir.parent / "configs" / "config.yaml")
        
    if not os.path.exists(config_path):
        return Config({})
        
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f) or {}
    return Config(config_dict)
