import subprocess
import os
from tqdm import tqdm

configs = [
    "../configs/datagen/datagen_train_val_mixed.toml",
    "../configs/datagen/datagen_train_val_static.toml",
    "../configs/datagen/datagen_test_static_singlesource.toml",
    "../configs/datagen/datagen_test_dynamic_singlesource.toml",
]

# use logic cores / 2 for n_jobs
cpu_count = os.cpu_count()
if cpu_count is None:
    n_jobs = 1
else:
    n_jobs = cpu_count // 2

for config in tqdm(configs):
    print(f"Processing config: {config}")
    print("Step 1: Generate TASCAR projects")

    p = subprocess.Popen(
        ["python3", "01_generate_tascar_projects.py", "-c", config, "-j", str(n_jobs)]
    )
    exit_code = p.wait()
    if exit_code != 0:
        print(f"Error in processing config {config}. Exit code: {exit_code}")
        break

    print("Step 2: Generate innovation signals")
    p = subprocess.Popen(
        ["python3", "02_generate_innovationsignals.py", "-c", config, "-j", str(n_jobs)]
    )
    exit_code = p.wait()
    if exit_code != 0:
        print(f"Error in processing config {config}. Exit code: {exit_code}")
        break

    print("Step 3: Render TASCAR scenes")
    p = subprocess.Popen(
        ["python3", "03_render_tascar_scenes.py", "-c", config, "-j", str(n_jobs)]
    )
    exit_code = p.wait()
    if exit_code != 0:
        print(f"Error in processing config {config}. Exit code: {exit_code}")
        break

    print("Step 4: Post-process audio files")
    p = subprocess.Popen(
        ["python3", "04_postprocess_audio.py", "-c", config, "-j", str(n_jobs)]
    )
    exit_code = p.wait()
    if exit_code != 0:
        print(f"Error in processing config {config}. Exit code: {exit_code}")
        break

    print(f"Completed processing for config: {config}\n\n")
