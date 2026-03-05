import os
from joblib import Parallel, delayed
import numpy as np
import soundfile as sf
import toml
from tqdm import tqdm
import subprocess
import argparse


def postprocess_audio(fn, start_idx):
    audio, fs = sf.read(fn)
    if np.max(np.abs(audio)) > 0.99:
        print(f"Warning: Clipping detected in file {fn}. Max value: {np.max(np.abs(audio))}")
        
    sf.write(fn, audio[start_idx:], fs, "PCM_32")


parser = argparse.ArgumentParser()
parser.add_argument(
    "-c",
    "--config",
    nargs="?",
    const=1,
    type=str,
    default="../configs/datagen_train_val_mixed.toml",
    help="Path to config file",
)
parser.add_argument(
    "-j",
    "--n_jobs",
    nargs="?",
    const=1,
    type=int,
    default=24,
    help="Number of parallel jobs",
)

args = parser.parse_args()
config = toml.load(args.config)
n_jobs = int(args.n_jobs)

# Limit number of jobs to available CPUs
if not isinstance(n_jobs, int) or n_jobs < 1:
    n_jobs = os.cpu_count() // 2

n_jobs = min(n_jobs, os.cpu_count())


start_idx = int(config["data"]["offset"] * config["samplerate"])

N_SCENES_TRAIN = int(config["data"]["nscenes"] * config["data"]["train_val_test"][0])
N_SCENES_VALID = int(config["data"]["nscenes"] * config["data"]["train_val_test"][1])
N_SCENES_TEST = int(config["data"]["nscenes"] * config["data"]["train_val_test"][2])

if N_SCENES_TRAIN > 0:
    fn_prefix = config["data"]["directory"] + "/train/scene_"
    Parallel(n_jobs=n_jobs)(
        delayed(postprocess_audio)(fn_prefix + str(idx) + ".wav", start_idx)
        for idx in tqdm(range(N_SCENES_TRAIN), leave=False)
    )
    subprocess.run(
        "rm -rf " + config["data"]["directory"] + "/train/innovation_signals",
        shell=True,
    )

if N_SCENES_VALID > 0:
    fn_prefix = config["data"]["directory"] + "/valid/scene_"
    Parallel(n_jobs=n_jobs)(
        delayed(postprocess_audio)(fn_prefix + str(idx) + ".wav", start_idx)
        for idx in tqdm(range(N_SCENES_VALID), leave=False)
    )
    subprocess.run(
        "rm -rf " + config["data"]["directory"] + "/valid/innovation_signals",
        shell=True,
    )

if N_SCENES_TEST > 0:
    fn_prefix = config["data"]["directory"] + "/test/scene_"
    Parallel(n_jobs=n_jobs)(
        delayed(postprocess_audio)(fn_prefix + str(idx) + ".wav", start_idx)
        for idx in tqdm(range(N_SCENES_TEST), leave=False)
    )

    fn_prefix = config["data"]["directory"] + "/test/innovation_signals/innov_"
    Parallel(n_jobs=128)(
        delayed(postprocess_audio)(fn_prefix + str(idx) + ".wav", start_idx)
        for idx in tqdm(range(N_SCENES_TEST), leave=False)
    )
