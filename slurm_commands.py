import os

for file in os.listdir(os.getcwd()):
    if file.endswith(".py"):
        print(file)
        for partition in ["cpu","gpu"]:
            print(f"sbatch -J test --err=slurm_chip/test/{file}_{partition}.err --out=slurm_chip/test/{file}_{partition}.out runpy{partition}_chip.sh {file}")
            print(f"sbatch -J test --err=slurm_chip/test/{file}_{partition}.err --out=slurm_chip/test/{file}_{partition}.out runpy{partition}_chip.sh {file}")
        print("")