import subprocess
from tqdm import tqdm

configs = [
    "./configs/mod_D6_S3_L32_II32.toml",
    "./configs/orig_D6_S3_L32_II32.toml",
    "./configs/mod_D6_S3_L32_II32_Ctcn512.toml",
]

for config in tqdm(configs):
    print(f"Processing config: {config}")

    p = subprocess.Popen(["python3", "test_and_export.py", "-c", config])
    exit_code = p.wait()
    if exit_code != 0:
        print(f"Error in processing config {config}. Exit code: {exit_code}")
        break
