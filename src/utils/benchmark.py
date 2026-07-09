import os
import sys
import time
import json
import platform
import subprocess
from datetime import datetime

def get_system_metadata() -> dict:
    metadata = {
        "os": platform.system(),
        "cpu": "Unknown",
        "ram_gb": 8,
        "gpu": "None"
    }
    
    try:
        if platform.system() == "Darwin":
            cpu = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"]).decode().strip()
            metadata["cpu"] = cpu
        elif platform.system() == "Linux":
            with open("/proc/cpuinfo", "r") as f:
                for line in f:
                    if "model name" in line:
                        metadata["cpu"] = line.split(":")[1].strip()
                        break
    except Exception:
        pass
        
    try:
        if platform.system() == "Darwin":
            mem_bytes = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"]).decode().strip())
            metadata["ram_gb"] = round(mem_bytes / (1024 ** 3))
        elif platform.system() == "Linux":
            with open("/proc/meminfo", "r") as f:
                for line in f:
                    if "MemTotal" in line:
                        mem_kb = int(line.split()[1])
                        metadata["ram_gb"] = round(mem_kb / (1024 ** 2))
                        break
    except Exception:
        pass
        
    try:
        import torch
        if torch.cuda.is_available():
            metadata["gpu"] = torch.cuda.get_device_name(0)
        elif torch.backends.mps.is_available():
            metadata["gpu"] = "Apple Metal (MPS)"
    except Exception:
        pass
        
    return metadata

def run_benchmark():
    os.makedirs("reports", exist_ok=True)
    timestamp = datetime.utcnow().isoformat() + "Z"
    date_str = datetime.utcnow().strftime("%Y-%m-%d")
    
    report = {
        "project": "VisionGuard AI",
        "timestamp": timestamp,
        "status": "not_run",
        "hardware": get_system_metadata(),
        "environment": {
            "python": platform.python_version()
        },
        "metadata": {
            "model": "U-Net",
            "parameters_million": 31.0,
            "dataset": "Synthetic Brain MRI",
            "batch_size": 1,
            "image_size": 256,
            "sequence_length": None,
            "device": "cpu"
        },
        "benchmarks": {}
    }
    
    # Check dependencies first
    required_libs = ["torch", "onnxruntime", "transformers", "albumentations"]
    missing_deps = []
    
    for lib in required_libs:
        try:
            mod = __import__(lib)
            report["environment"][lib] = getattr(mod, "__version__", "installed")
        except ImportError:
            missing_deps.append(lib)
            
    if missing_deps:
        report["status"] = "not_run"
        report["reason"] = f"Missing required dependencies: {', '.join(missing_deps)}"
        report["required_dependency"] = missing_deps[0]
        
        save_reports(report, date_str)
        print(f"Benchmark not run: {report['reason']}")
        return
        
    # Run actual benchmarks inside try to prevent import errors at module loading
    try:
        import torch
        import onnxruntime as ort
        from src.models.unet import UNet
        
        device = torch.device("cpu")
        model = UNet(in_channels=1, num_classes=2)
        model.eval()
        
        dummy_input = torch.randn(1, 1, 256, 256)
        
        print("Benchmarking PyTorch UNet on CPU...")
        start_time = time.time()
        for _ in range(20):
            with torch.no_grad():
                _ = model(dummy_input)
        pytorch_latency = ((time.time() - start_time) / 20) * 1000
        
        config_path = "reports/model.onnx"
        if not os.path.exists(config_path):
            torch.onnx.export(
                model, dummy_input, config_path,
                input_names=['input'], output_names=['output']
            )
            
        print("Benchmarking ONNX Runtime on CPU...")
        ort_session = ort.InferenceSession(config_path, providers=['CPUExecutionProvider'])
        input_name = ort_session.get_inputs()[0].name
        numpy_input = dummy_input.numpy()
        
        start_time = time.time()
        for _ in range(20):
            _ = ort_session.run(None, {input_name: numpy_input})
        onnx_latency = ((time.time() - start_time) / 20) * 1000
        
        speedup = pytorch_latency / onnx_latency
        
        report["status"] = "success"
        report["benchmarks"] = {
            "pytorch_cpu_latency_ms": round(pytorch_latency, 2),
            "onnx_cpu_latency_ms": round(onnx_latency, 2),
            "speedup_factor": round(speedup, 2),
            "pytorch_fps": round(1000 / pytorch_latency, 2),
            "onnx_fps": round(1000 / onnx_latency, 2)
        }
        
        print(f"Benchmark success: PyTorch = {pytorch_latency:.2f}ms, ONNX = {onnx_latency:.2f}ms, Speedup = {speedup:.2f}x")
    except Exception as e:
        report["status"] = "error"
        report["reason"] = f"Benchmark execution error: {e}"
        print(f"Benchmark error: {e}")
        
    save_reports(report, date_str)

def save_reports(report: dict, date_str: str):
    with open(f"reports/{date_str}-benchmark.json", "w") as f:
        json.dump(report, f, indent=4)
    with open("reports/latest.json", "w") as f:
        json.dump(report, f, indent=4)

if __name__ == "__main__":
    run_benchmark()
