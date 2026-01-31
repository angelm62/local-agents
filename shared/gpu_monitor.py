"""
GPU Monitoring Utilities
Real-time GPU usage monitoring using nvidia-ml-py3 and GPUtil
"""

import time
import os
from typing import Optional, List, Dict, Any


def check_nvidia_smi() -> bool:
    """Check if nvidia-smi is available"""
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi'], capture_output=True, timeout=5)
        return result.returncode == 0
    except Exception:
        return False


def get_gpu_info_torch() -> Dict[str, Any]:
    """Get GPU info using PyTorch"""
    try:
        import torch

        if not torch.cuda.is_available():
            return {"available": False, "message": "CUDA not available in PyTorch"}

        info = {
            "available": True,
            "device_count": torch.cuda.device_count(),
            "devices": []
        }

        for i in range(torch.cuda.device_count()):
            device_info = {
                "id": i,
                "name": torch.cuda.get_device_name(i),
                "memory_allocated_gb": torch.cuda.memory_allocated(i) / 1024**3,
                "memory_reserved_gb": torch.cuda.memory_reserved(i) / 1024**3,
                "memory_total_gb": torch.cuda.get_device_properties(i).total_memory / 1024**3,
            }
            info["devices"].append(device_info)

        return info

    except Exception as e:
        return {"available": False, "error": str(e)}


def get_gpu_info_nvml() -> Dict[str, Any]:
    """Get detailed GPU info using NVIDIA Management Library"""
    try:
        import pynvml

        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()

        info = {
            "available": True,
            "device_count": device_count,
            "driver_version": pynvml.nvmlSystemGetDriverVersion(),
            "devices": []
        }

        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)

            # Get device info
            name = pynvml.nvmlDeviceGetName(handle)
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert to watts

            device_info = {
                "id": i,
                "name": name,
                "memory_total_gb": memory_info.total / 1024**3,
                "memory_used_gb": memory_info.used / 1024**3,
                "memory_free_gb": memory_info.free / 1024**3,
                "memory_used_percent": (memory_info.used / memory_info.total) * 100,
                "gpu_utilization_percent": utilization.gpu,
                "memory_utilization_percent": utilization.memory,
                "temperature_c": temperature,
                "power_usage_w": power,
            }
            info["devices"].append(device_info)

        pynvml.nvmlShutdown()
        return info

    except Exception as e:
        return {"available": False, "error": str(e)}


def get_gpu_info_gputil() -> Dict[str, Any]:
    """Get GPU info using GPUtil"""
    try:
        import GPUtil

        gpus = GPUtil.getGPUs()

        if not gpus:
            return {"available": False, "message": "No GPUs detected by GPUtil"}

        info = {
            "available": True,
            "device_count": len(gpus),
            "devices": []
        }

        for gpu in gpus:
            device_info = {
                "id": gpu.id,
                "name": gpu.name,
                "memory_total_gb": gpu.memoryTotal / 1024,
                "memory_used_gb": gpu.memoryUsed / 1024,
                "memory_free_gb": gpu.memoryFree / 1024,
                "memory_used_percent": gpu.memoryUtil * 100,
                "gpu_utilization_percent": gpu.load * 100,
                "temperature_c": gpu.temperature,
            }
            info["devices"].append(device_info)

        return info

    except Exception as e:
        return {"available": False, "error": str(e)}


def print_gpu_info(detailed: bool = True):
    """Print GPU information in a formatted way"""
    print("=" * 70)
    print("GPU Information")
    print("=" * 70)

    # Try NVML first (most detailed)
    info = get_gpu_info_nvml()

    if not info.get("available"):
        # Fallback to GPUtil
        print("NVML not available, trying GPUtil...")
        info = get_gpu_info_gputil()

    if not info.get("available"):
        # Fallback to PyTorch
        print("GPUtil not available, trying PyTorch...")
        info = get_gpu_info_torch()

    if not info.get("available"):
        print("No GPU information available")
        print(f"Error: {info.get('error', info.get('message', 'Unknown error'))}")
        return

    # Print summary
    print(f"GPUs Detected: {info['device_count']}")
    if "driver_version" in info:
        print(f"Driver Version: {info['driver_version']}")
    print()

    # Print device details
    for device in info["devices"]:
        print(f"GPU {device['id']}: {device['name']}")
        print(f"  Memory: {device['memory_used_gb']:.2f} GB / {device['memory_total_gb']:.2f} GB "
              f"({device.get('memory_used_percent', 0):.1f}% used)")

        if detailed:
            if "memory_free_gb" in device:
                print(f"  Free Memory: {device['memory_free_gb']:.2f} GB")
            if "gpu_utilization_percent" in device:
                print(f"  GPU Utilization: {device['gpu_utilization_percent']:.1f}%")
            if "memory_utilization_percent" in device:
                print(f"  Memory Utilization: {device['memory_utilization_percent']:.1f}%")
            if "temperature_c" in device:
                print(f"  Temperature: {device['temperature_c']}°C")
            if "power_usage_w" in device:
                print(f"  Power Usage: {device['power_usage_w']:.1f} W")
        print()

    print("=" * 70)


def monitor_gpu(interval: int = 2, duration: Optional[int] = None):
    """
    Monitor GPU usage in real-time

    Args:
        interval: Update interval in seconds
        duration: Total monitoring duration in seconds (None for infinite)
    """
    try:
        start_time = time.time()
        iteration = 0

        print("Starting GPU monitoring... (Press Ctrl+C to stop)")
        print()

        while True:
            # Clear screen (works on Unix-like systems)
            os.system('clear' if os.name != 'nt' else 'cls')

            print(f"GPU Monitor - Iteration {iteration + 1}")
            print(f"Elapsed: {time.time() - start_time:.1f}s")
            print()

            print_gpu_info(detailed=True)

            iteration += 1

            # Check duration
            if duration and (time.time() - start_time) >= duration:
                print(f"Monitoring complete after {duration} seconds")
                break

            time.sleep(interval)

    except KeyboardInterrupt:
        print("\nMonitoring stopped by user")


def get_optimal_device() -> str:
    """
    Determine the optimal device for computation

    Returns:
        "cuda" if GPU available and has free memory, else "cpu"
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return "cpu"

        # Check if GPU has sufficient free memory (at least 1GB)
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            allocated = torch.cuda.memory_allocated(i)
            free = props.total_memory - allocated

            if free > 1024**3:  # 1GB
                return f"cuda:{i}"

        return "cpu"

    except Exception:
        return "cpu"


if __name__ == "__main__":
    import sys

    # Check if monitoring mode requested
    if len(sys.argv) > 1 and sys.argv[1] == "monitor":
        interval = int(sys.argv[2]) if len(sys.argv) > 2 else 2
        duration = int(sys.argv[3]) if len(sys.argv) > 3 else None
        monitor_gpu(interval=interval, duration=duration)
    else:
        # Just print current info
        print_gpu_info(detailed=True)

        # Show optimal device
        print()
        optimal = get_optimal_device()
        print(f"Optimal Device: {optimal}")
