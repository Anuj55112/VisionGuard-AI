import os
import time
import json
import torch
from pathlib import Path
from src.config import load_config
from src.models.unet import UNet

def export_unet_to_onnx():
    config = load_config()
    os.makedirs(os.path.dirname(config.onnx_export_path), exist_ok=True)
    
    print("Initializing PyTorch UNet model...")
    model = UNet(
        in_channels=config.unet_in_channels,
        num_classes=config.unet_num_classes,
        bilinear=config.unet_bilinear
    )
    model.eval()
    
    # Create dummy input: [Batch, Channel, Height, Width]
    dummy_input = torch.randn(1, config.unet_in_channels, config.image_size, config.image_size)
    
    print(f"Exporting model to ONNX format at {config.onnx_export_path}...")
    torch.onnx.export(
        model,
        dummy_input,
        config.onnx_export_path,
        export_params=True,
        opset_version=config.onnx_opset_version,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print("ONNX model exported successfully!")
    
    # Run latency benchmarks
    benchmark_results = run_benchmark(model, config.onnx_export_path, dummy_input)
    
    # Save benchmark results
    benchmark_file = Path(config.onnx_export_path).parent / "onnx_benchmark.json"
    with open(benchmark_file, "w") as f:
        json.dump(benchmark_results, f, indent=4)
    print(f"Benchmark results saved to {benchmark_file}")

def run_benchmark(pytorch_model, onnx_path, dummy_input) -> dict:
    print("Running performance benchmarks...")
    results = {}
    
    # PyTorch CPU Benchmark
    pytorch_model.cpu()
    start_time = time.time()
    for _ in range(50):
        with torch.no_grad():
            _ = pytorch_model(dummy_input)
    results["pytorch_cpu_latency_ms"] = ((time.time() - start_time) / 50) * 1000
    
    # ONNX Runtime CPU Benchmark
    try:
        import onnxruntime as ort
        ort_session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        input_name = ort_session.get_inputs()[0].name
        numpy_input = dummy_input.numpy()
        
        start_time = time.time()
        for _ in range(50):
            _ = ort_session.run(None, {input_name: numpy_input})
        results["onnx_cpu_latency_ms"] = ((time.time() - start_time) / 50) * 1000
    except Exception as e:
        print(f"Failed to benchmark ONNX CPU: {e}")
        results["onnx_cpu_latency_ms"] = None
        
    print(f"PyTorch CPU Avg Latency: {results['pytorch_cpu_latency_ms']:.2f} ms")
    if results["onnx_cpu_latency_ms"]:
        print(f"ONNX CPU Avg Latency: {results['onnx_cpu_latency_ms']:.2f} ms")
        speedup = results['pytorch_cpu_latency_ms'] / results['onnx_cpu_latency_ms']
        print(f"ONNX Speedup: {speedup:.2fx}")
        results["speedup"] = speedup
        
    return results

if __name__ == "__main__":
    export_unet_to_onnx()
