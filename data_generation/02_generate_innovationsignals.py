import os
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
import colorednoise as cn
import soundfile as sf
import pickle
import toml
import argparse


def render_innovation_signals(config, subdir, idx, metadata):
    # Get number of required source signals
    n_src = metadata["src_n"]

    len_audio = (config["data"]["length"] + config["data"]["offset"]) * config[
        "samplerate"
    ]

    n_sines = np.random.randint(
        config["data"]["src"]["nsines"][0],
        config["data"]["src"]["nsines"][1] + 1,
        n_src,
    )

    noise_level = 10 ** (
        np.random.uniform(
            config["data"]["src"]["levelnoise"][0],
            config["data"]["src"]["levelnoise"][1],
            n_src,
        )
        / 20
    )

    noise_exponent = np.random.uniform(
        config["data"]["src"]["noiseexponent"][0],
        config["data"]["src"]["noiseexponent"][1],
        n_src,
    )

    overall_level = 10 ** (
        np.random.uniform(
            config["data"]["src"]["leveloverall"][0],
            config["data"]["src"]["leveloverall"][1],
            n_src,
        )
        / 20
    )

    innovation_signals = np.zeros((n_src, len_audio))

    for src_idx in range(n_src):

        innovation_signals[src_idx] += noise_level[src_idx] * cn.powerlaw_psd_gaussian(
            noise_exponent[src_idx], len_audio
        )

        # Generate sine signals
        if n_sines[src_idx] > 0:
            sine_frequencies = np.random.uniform(
                config["data"]["src"]["sinefrequency"][0],
                config["data"]["src"]["sinefrequency"][1],
                n_sines[src_idx],
            )
            sine_levels = 10 ** (
                np.random.uniform(
                    config["data"]["src"]["levelsines"][0],
                    config["data"]["src"]["levelsines"][1],
                    n_sines[src_idx],
                )
                / 20
            )

            t = np.arange(len_audio) / config["samplerate"]
            for i in range(n_sines[src_idx]):
                sine_signal = sine_levels[i] * np.sin(
                    2 * np.pi * sine_frequencies[i] * t
                )
                innovation_signals[src_idx] += sine_signal

        # Normalize the innovation signal and apply gain
        innovation_signals[src_idx] *= overall_level[src_idx] / np.sqrt(
            np.mean(innovation_signals[src_idx] ** 2)
        )

    if np.max(np.abs(innovation_signals)) > 1.0:
        print(
            f"Warning: Clipping in innovation signals for scene {idx} in {subdir}. Max value: {np.max(np.abs(innovation_signals))}"
        )

    # Save innovation signals
    fn = (
        config["data"]["directory"]
        + subdir
        + "innovation_signals/innov_"
        + str(idx)
        + ".wav"
    )
    sf.write(fn, innovation_signals.T, config["samplerate"], subtype="PCM_32")

    metadata["src_n_sines"] = n_sines.tolist()
    metadata["src_noise_level"] = noise_level.tolist()
    metadata["src_noise_exponent"] = noise_exponent.tolist()
    metadata["src_level"] = overall_level.tolist()

    return metadata


# Load configuration
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

# Load metadata
if config["data"]["train_val_test"][0] > 0:
    with open(config["data"]["tscdirectory"] + "/train/metadata.pkl", "rb") as f:
        metadata_train = pickle.load(f)
if config["data"]["train_val_test"][1] > 0:
    with open(config["data"]["tscdirectory"] + "/valid/metadata.pkl", "rb") as f:
        metadata_valid = pickle.load(f)
if config["data"]["train_val_test"][2] > 0:
    with open(config["data"]["tscdirectory"] + "/test/metadata.pkl", "rb") as f:
        metadata_test = pickle.load(f)

# Split datasets
N_SCENES_TRAIN = int(config["data"]["nscenes"] * config["data"]["train_val_test"][0])
N_SCENES_VALID = int(config["data"]["nscenes"] * config["data"]["train_val_test"][1])
N_SCENES_TEST = int(config["data"]["nscenes"] * config["data"]["train_val_test"][2])

# Random seed
np.random.seed(config["data"]["seed"])

# Create subdirectories for innovation signals
if N_SCENES_TRAIN > 0:
    os.makedirs(
        config["data"]["directory"] + "/train/innovation_signals", exist_ok=False
    )
    metadata_train_new = Parallel(n_jobs=n_jobs)(
        delayed(render_innovation_signals)(config, "/train/", idx, metadata)
        for idx, metadata in tqdm(
            enumerate(metadata_train), total=N_SCENES_TRAIN, leave=False
        )
    )
    ordered_metadata = sorted(metadata_train_new, key=lambda x: x["index"])
    with open(config["data"]["directory"] + "/train/metadata.pkl", "wb") as f:
        pickle.dump(ordered_metadata, f)

if N_SCENES_VALID > 0:
    os.makedirs(
        config["data"]["directory"] + "/valid/innovation_signals", exist_ok=False
    )
    metadata_valid_new = Parallel(n_jobs=n_jobs)(
        delayed(render_innovation_signals)(config, "/valid/", idx, metadata)
        for idx, metadata in tqdm(
            enumerate(metadata_valid), total=N_SCENES_VALID, leave=False
        )
    )
    ordered_metadata = sorted(metadata_valid_new, key=lambda x: x["index"])
    with open(config["data"]["directory"] + "/valid/metadata.pkl", "wb") as f:
        pickle.dump(ordered_metadata, f)

if N_SCENES_TEST > 0:
    os.makedirs(
        config["data"]["directory"] + "/test/innovation_signals", exist_ok=False
    )
    metadata_test_new = Parallel(n_jobs=n_jobs)(
        delayed(render_innovation_signals)(config, "/test/", idx, metadata)
        for idx, metadata in tqdm(
            enumerate(metadata_test), total=N_SCENES_TEST, leave=False
        )
    )
    ordered_metadata = sorted(metadata_test_new, key=lambda x: x["index"])
    with open(config["data"]["directory"] + "/test/metadata.pkl", "wb") as f:
        pickle.dump(ordered_metadata, f)

    # Create a single impulse for rendering IRs
    impulse = np.zeros((int(config["samplerate"]), 2))
    impulse[0, 1] = 1.0

    sf.write(
        config["data"]["directory"] + "/test/impulse.wav",
        impulse,
        config["samplerate"],
        subtype="PCM_32",
    )
