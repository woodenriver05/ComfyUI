import time
import torch
import subprocess

def run_comfy_workflow(device):
    print(f"\n🔁 Running on {device.upper()}...")
    torch_device = torch.device("mps" if device == "mps" else "cpu")
    torch.set_default_device(torch_device)

    start = time.time()
    subprocess.run([
        "python", "main.py",
        "--workflow", "input/benchmark_sdxl.json"
    ])
    end = time.time()

    print(f"✅ {device.upper()} run completed in {end - start:.2f} seconds")

    with open("benchmark_results.txt", "a") as f:
        f.write(f"{device.upper()} run: {end - start:.2f} seconds\n")

run_comfy_workflow("mps")
run_comfy_workflow("cpu")
 
