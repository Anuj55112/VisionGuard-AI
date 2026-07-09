import pytest
import io
from fastapi.testclient import TestClient
from PIL import Image
import numpy as np

from app.api import app

client = TestClient(app)

def test_api_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"

def test_api_segmentation_prediction():
    # Create sample image in bytes
    img = Image.fromarray(np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8))
    img_io = io.BytesIO()
    img.save(img_io, 'PNG')
    img_io.seek(0)
    
    files = {"file": ("image.png", img_io, "image/png")}
    response = client.post("/predict", files=files, data={"model_type": "unet"})
    
    assert response.status_code == 200
    # Response is streaming PNG mask
    assert response.headers["content-type"] == "image/png"
    
    # Verify we can open returned mask
    mask = Image.open(io.BytesIO(response.content))
    assert mask.size == (128, 128)

def test_api_sam_prediction():
    img = Image.fromarray(np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8))
    img_io = io.BytesIO()
    img.save(img_io, 'PNG')
    img_io.seek(0)
    
    files = {"file": ("image.png", img_io, "image/png")}
    response = client.post("/sam/predict", files=files, data={"points": "[[50, 50]]"})
    
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
