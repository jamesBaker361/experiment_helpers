import os
import torch

def print_details():
    for slurm_var in ["SLURMD_NODENAME","SBATCH_CLUSTERS", 
                      "SBATCH_PARTITION","SLURM_JOB_PARTITION",
                      "SLURM_NODEID","SLURM_MEM_PER_GPU",
                      "SLURM_MEM_PER_CPU","SLURM_MEM_PER_NODE","SLURM_JOB_ID"]:
        try:
            print(slurm_var, os.environ[slurm_var])
        except:
            print(slurm_var, "doesnt exist")
    try:
        print('torch.cuda.get_device_name()',torch.cuda.get_device_name())
        print('torch.cuda.get_device_capability()',torch.cuda.get_device_capability())
        current_device = torch.cuda.current_device()
        gpu = torch.cuda.get_device_properties(current_device)
        print(f"GPU Name: {gpu.name}")
        print(f"GPU Memory Total: {gpu.total_memory / 1024**2} MB")
        print(f"GPU Memory Free: {torch.cuda.memory_allocated(current_device) / 1024**2} MB")
        print(f"GPU Memory Used: {torch.cuda.memory_reserved(current_device) / 1024**2} MB")
    except:
        print("couldnt print cuda details")
    print("Torch:", torch.__version__)
    print("CUDA version:", torch.version.cuda)
    print("CUDA available:", torch.cuda.is_available())
    
    x = torch.randn(1, 3, 32, 32)
    conv = torch.nn.Conv2d(3, 4, 3)

    try:
        y = conv(x)
        print("Conv2d works on CPU")
    except Exception as e:
        print("CPU conv failed:", e)

    if torch.cuda.is_available():
        conv = conv.cuda()
        x = x.cuda()
        try:
            y = conv(x)
            print("Conv2d works on CUDA")
        except Exception as e:
            print("CUDA conv failed:", e)