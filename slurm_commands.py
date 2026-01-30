import os

for file in os.listdir(os.getcwd()):
    if file.endswith(".py"):
        print(f"sbatch -J test --err=slurm_chip/test/{file}.err --out=slurm_chip/test/{file}.out runpycpu_chip.sh {file}")
        print(f"sbatch -J test --err=slurm_chip/test/{file}.err --out=slurm_chip/test/{file}.out runpygpu_chip.sh {file}")