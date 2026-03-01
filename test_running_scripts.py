import os
import sys
import re
import subprocess
pattern = r"^run.*\.py$"

for file in os.listdir(os.getcwd()):
    if file.startswith("run") and file.endswith(".sh"):
        base = os.path.splitext(file)[0]  # removes .py
        command = [
            "sbatch",
            "-J", "runtest",
            "--err=slurm_chip/runtest/{}.err".format(base),
            "--out=slurm_chip/runtest/{}.out".format(base),
            file,
            "experiment_helpers/dummy.py"
        ]
        
        print(command)
        
        subprocess.run(command)