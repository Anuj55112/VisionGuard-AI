import torch
import numpy as np
from PIL import Image
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class SAMHelper:
    """
    Helper wrapper for the Segment Anything Model (SAM) using HuggingFace Transformers.
    Allows zero-shot interactive segmentation based on point prompts.
    """
    def __init__(self, model_name: str = "facebook/sam-vit-base"):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        self.model = None
        self.processor = None
        self.initialized = False
        
    def initialize(self) -> bool:
        if self.initialized:
            return True
        try:
            from transformers import SamModel, SamProcessor
            logger.info(f"Loading SAM weights for {self.model_name}...")
            self.processor = SamProcessor.from_pretrained(self.model_name)
            self.model = SamModel.from_pretrained(self.model_name).to(self.device)
            self.initialized = True
            logger.info("SAM model loaded successfully.")
            return True
        except Exception as e:
            logger.warning(f"Could not load SAM weights: {e}. Interactive SAM will run in mock mode.")
            return False

    def predict_mask(
        self,
        image: Image.Image,
        input_points: List[Tuple[int, int]],
        input_labels: Optional[List[int]] = None
    ) -> np.ndarray:
        if not self.initialized:
            success = self.initialize()
            if not success or self.model is None or self.processor is None:
                return self._mock_prediction(image, input_points)
                
        try:
            if input_labels is None:
                input_labels = [1] * len(input_points)
                
            points_formatted = [[input_points]]
            labels_formatted = [[input_labels]]
            
            inputs = self.processor(
                image,
                input_points=points_formatted,
                input_labels=labels_formatted,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                
            masks = self.processor.image_processor.post_process_masks(
                outputs.pred_masks.cpu(),
                inputs["original_sizes"].cpu(),
                inputs["reshaped_input_sizes"].cpu()
            )
            
            binary_mask = masks[0][0][0].numpy()
            return binary_mask.astype(np.uint8)
            
        except Exception as e:
            logger.error(f"Error during SAM inference: {e}")
            return self._mock_prediction(image, input_points)

    def _mock_prediction(self, image: Image.Image, input_points: List[Tuple[int, int]]) -> np.ndarray:
        w, h = image.size
        mask = np.zeros((h, w), dtype=np.uint8)
        y_grid, x_grid = np.ogrid[:h, :w]
        for pt in input_points:
            pt_x, pt_y = pt[0], pt[1]
            dist = (x_grid - pt_x) ** 2 + (y_grid - pt_y) ** 2
            mask[dist <= 40**2] = 1
        return mask
