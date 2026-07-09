# Model Card: VisionGuard AI Segmentation Engines

This model card documents the specifications and details of the segmentation architectures deployed within the **VisionGuard AI** platform.

## Model Details
- **Developer**: Portfolio Owner
- **Model Types**:
  - **U-Net**: Classic convolutional encoder-decoder network with skip connections for high-resolution detail preservation.
  - **SegFormer**: Transformer-based encoder (`nvidia/mit-b0`) coupled with a lightweight MLP decoder for semantic segmentation.
  - **Segment Anything Model (SAM)**: Zero-shot foundation model (`facebook/sam-vit-base`) triggered dynamically via click coordinates.
- **Task**: Binary / Multi-class Medical Image Segmentation (MRI & Chest X-Ray scan boundaries).

## Intended Use
- **Primary Intended Use**: Research, benchmarking, and demonstration of comparative performance between convolutional neural networks, vision transformers, and interactive foundation models.
- **Out of Scope Use**: Direct diagnostic use. These models are built for benchmarking and portfolio presentation, not clinical diagnostic applications without clinical supervision.

## Datasets & Training
- **Training Data**: Evaluated using medical segmentation targets (e.g. Brain Tumor Segmentation or synthetic MRI/X-ray equivalents).
- **Preprocessing**:
  - Images resized to 256x256.
  - Pixel values normalized using ImageNet mean `(0.485, 0.456, 0.406)` and standard deviation `(0.229, 0.224, 0.225)`.
  - Data Augmentations: Random rotate, horizontal/vertical flips, elastic transforms, shift-scale-rotate.

## Evaluation & Metrics
- **Primary Metric**: Dice Similarity Coefficient (F1-Score).
- **Latency Benchmarks**: Latency computed on standard CPU runtime using FP32 precision compared against ONNX optimized runtimes.

## Model Limitations & Bias
- Under-performs on images with low contrast ratios.
- Intended for demonstration. Checkpoint weights require full training on task-specific segmentation datasets (e.g. BraTS, COVID-QU-Ex) before target deployment.
