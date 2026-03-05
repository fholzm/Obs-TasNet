import os
import subprocess
from multiprocessing import Pool, Manager
from tqdm import tqdm
import toml
import argparse


def render_tascar_scenes(config, subdir, idx, logfile):
    try:
        absolute_path_innov = os.path.join(
            config["data"]["directory"]
            + subdir
            + "innovation_signals/innov_"
            + str(idx)
            + ".wav"
        )
        absolute_path_tsc = os.path.join(
            "/workspaces/deep-observationfilter/data_generation/",
            config["data"]["tscdirectory"][2:]
            + subdir
            + "deep_observationfilter_scene_"
            + str(idx)
            + ".tsc",
        )
        cmd = (
            "tascar_renderfile --verbose --srate "
            + str(config["samplerate"])
            + " --inputfile "
            + absolute_path_innov
            + " --outputfile "
            + config["data"]["directory"]
            + subdir
            + "scene_"
            + str(idx)
            + ".wav "
            + absolute_path_tsc
        )
        result = subprocess.run(
            cmd, shell=True, check=True, text=True, capture_output=True
        )

        with open(logfile, "a") as log:
            log.write(f"Scene {idx} rendered successfully.\n")
            log.write(result.stdout)
            log.write(result.stderr)
            log.write("\n")

        if subdir == "/test/":
            absolute_impulse_path = os.path.join(
                config["data"]["directory"] + subdir + "impulse.wav"
            )

            cmd = (
                "tascar_renderfile --verbose --srate "
                + str(config["samplerate"])
                + " --inputfile "
                + absolute_impulse_path
                + " --outputfile "
                + config["data"]["directory"]
                + subdir
                + "ir_"
                + str(idx)
                + ".wav "
                + absolute_path_tsc
            )
            result = subprocess.run(
                cmd, shell=True, check=True, text=True, capture_output=True
            )

            with open(logfile, "a") as log:
                log.write(f"Scene {idx} rendered successfully.\n")
                log.write(result.stdout)
                log.write(result.stderr)
                log.write("\n")

    except subprocess.CalledProcessError as e:
        with open(logfile, "a") as log:
            log.write(f"Error rendering scene {idx}:\n")
            log.write(str(e))
            log.write(e.output)
            log.write("\n")


def run_rendering(args):
    config, subdir, idx, logfile = args
    render_tascar_scenes(config, subdir, idx, logfile)


parser = argparse.ArgumentParser()
parser.add_argument(
    "-c",
    "--config",
    nargs="?",
    const=1,
    type=str,
    default="../configs/datagen/datagen_train_val_mixed.toml",
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
config["data"]["directory"] = os.path.abspath(config["data"]["directory"])
n_jobs = int(args.n_jobs)

# Limit number of jobs to available CPUs
if not isinstance(n_jobs, int) or n_jobs < 1:
    n_jobs = os.cpu_count() // 2

n_jobs = min(n_jobs, os.cpu_count())

N_SCENES_TRAIN = int(config["data"]["nscenes"] * config["data"]["train_val_test"][0])
N_SCENES_VALID = int(config["data"]["nscenes"] * config["data"]["train_val_test"][1])
N_SCENES_TEST = int(config["data"]["nscenes"] * config["data"]["train_val_test"][2])

os.remove("*.log") if os.path.exists("*.log") else None

manager = Manager()
progress_queue = manager.Queue()

# Render training data
if N_SCENES_TRAIN > 0:
    tasks = [
        (config, "/train/", idx, "output_train.log") for idx in range(N_SCENES_TRAIN)
    ]

    with tqdm(total=N_SCENES_TRAIN, leave=False) as pbar:
        with Pool(processes=n_jobs) as pool:
            for _ in pool.imap_unordered(run_rendering, tasks):
                pbar.update(1)

# Render validation data
if N_SCENES_VALID > 0:
    tasks = [
        (config, "/valid/", idx, "output_valid.log") for idx in range(N_SCENES_VALID)
    ]

    with tqdm(total=N_SCENES_VALID, leave=False) as pbar:
        with Pool(processes=n_jobs) as pool:
            for _ in pool.imap_unordered(run_rendering, tasks):
                pbar.update(1)

# Render test data
if N_SCENES_TEST > 0:
    tasks = [(config, "/test/", idx, "output_test.log") for idx in range(N_SCENES_TEST)]

    with tqdm(total=N_SCENES_TEST, leave=False) as pbar:
        with Pool(processes=n_jobs) as pool:
            for _ in pool.imap_unordered(run_rendering, tasks):
                pbar.update(1)
