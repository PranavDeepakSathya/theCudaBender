import subprocess

cmd = [
    "compute-sanitizer",
    "--tool", "memcheck",   # best for misaligned address bugs
    "--leak-check", "full",
    "python",
    "run.py"
]

with open("sanitizer_output.txt", "w") as f:
    subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)