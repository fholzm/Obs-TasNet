import os
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
import pickle
import toml
import csv
from lxml import etree, objectify
import argparse


def sph2cart(az, el, r):
    x = r * np.cos(az) * np.cos(el)
    y = r * np.sin(az) * np.cos(el)
    z = r * np.sin(el)
    return x, y, z


def randompos(config):
    pos_sph = np.random.uniform(
        [-np.pi, -np.pi / 2, config["data"]["vmic"]["range"][0]],
        [np.pi, np.pi / 2, config["data"]["vmic"]["range"][1]],
        3,
    )

    return np.array(config["data"]["vmic"]["centerpos"]).T + np.array(
        sph2cart(pos_sph[0], pos_sph[1], pos_sph[2])
    )


def generate_tascar_project(config, subdir, idx):
    # Create scene root
    root = objectify.Element(
        "session",
        attribution="Felix Holzmüller (autogen)",
    )
    scene = objectify.SubElement(
        root,
        "scene",
        name=f"deep_observationfilter_scene_{idx}",
        guiscale="6",
        ismorder="0",
    )

    # Add primary sources
    n_src = np.random.randint(
        config["data"]["src"]["nsrc"][0],
        config["data"]["src"]["nsrc"][1] + 1,
    )

    # Randomize source direction and distance
    src_sph = np.random.uniform(
        [-np.pi, -np.pi / 2, config["data"]["src"]["distance"][0]],
        [np.pi, np.pi / 2, config["data"]["src"]["distance"][1]],
        (n_src, 3),
    )

    # Get cartesian coordinates of sources
    src_pos = np.array(sph2cart(src_sph[:, 0], src_sph[:, 1], src_sph[:, 2])).T

    # Add primary sources
    for src_idx in range(n_src):
        source = objectify.SubElement(
            scene,
            "source",
            name="primarySource_" + str(src_idx),
        )

        x_pos = src_pos[src_idx, 0]
        y_pos = src_pos[src_idx, 1]
        z_pos = src_pos[src_idx, 2]

        # Add omnidrectional source with random gain
        # snd = objectify.SubElement(
        #     source, "sound", type="omni", gain=f"{src_gain[src_idx]:.3f}"
        # )
        snd = objectify.SubElement(source, "sound", type="omni")
        pos = objectify.SubElement(source, "position")._setText(
            f"0 {x_pos:.3f} {y_pos:.3f} {z_pos:.3f}"
        )

    # Add secondary source
    sec_src = objectify.SubElement(
        scene,
        "source",
        name="secondarySource",
    )

    objectify.SubElement(sec_src, "sound", type="omni", gain="-10.0")
    objectify.SubElement(sec_src, "position")._setText(
        f'0 {config["data"]["secsrc"]["position"][0]:.3f} {config["data"]["secsrc"]["position"][1]:.3f} {config["data"]["secsrc"]["position"][2]:.3f}'
    )

    # Add remote microphone array
    for micidx, miccoords in enumerate(config["data"]["rmic"]["position"]):
        rec = objectify.SubElement(
            scene,
            "receiver",
            name=f"rmic_{micidx}",
            type="omni",
            caliblevel=f'{config["data"]["rmic"]["caliblevel"]}',
        )

        rmic_position = np.array(miccoords) * config["data"]["rmic"]["scale"]

        # Set position
        objectify.SubElement(rec, "position")._setText(
            f"0 {rmic_position[0]:.3f} {rmic_position[1]:.3f} {rmic_position[2]:.3f}"
        )

    # Add virtual microphone
    rec = objectify.SubElement(
        scene,
        "receiver",
        name=f"vmic",
        type="omni",
        caliblevel=f'{config["data"]["vmic"]["caliblevel"]}',
    )

    objectify.SubElement(rec, "position", importcsv=f"vmic_pos_{idx}.csv")

    # Remove annotations from root
    objectify.deannotate(root, xsi_nil=True)
    etree.cleanup_namespaces(root)

    # Create string from xml
    obj_xml = etree.tostring(
        root, pretty_print=True, xml_declaration=True, encoding="utf-8"
    )

    # Write to file
    fn = (
        config["data"]["tscdirectory"]
        + subdir
        + f"deep_observationfilter_scene_{idx}.tsc"
    )

    try:
        with open(fn, "wb") as xml_writer:
            xml_writer.write(obj_xml)
    except IOError:
        pass

    # %% Write position data csv
    # Create timeline
    t_total = config["data"]["length"] + config["data"]["offset"]
    timeline = np.arange(
        0,
        t_total + 1 / config["data"]["samplerate_position"],
        1 / config["data"]["samplerate_position"],
    )

    # Define how many different paths are taken
    n_paths = np.random.randint(
        config["data"]["vmic"]["npos"][0] - 1,
        config["data"]["vmic"]["npos"][1],
    )

    # Generate start position
    start_position = randompos(config)
    coordinates = np.zeros((len(timeline), 3))

    if n_paths == 0:
        coordinates = np.repeat(start_position[np.newaxis, :], len(timeline), axis=0)
    else:
        # Add start with offset
        start = 0
        end = int(config["data"]["offset"] * config["data"]["samplerate_position"]) + 1
        coordinates[start:end] = start_position
        start = end

        # Add static part at begin
        end += (
            int(
                np.random.uniform(
                    config["data"]["vmic"]["initialpause"][0],
                    config["data"]["vmic"]["initialpause"][1],
                )
                * config["data"]["samplerate_position"]
            )
            + 1
        )
        coordinates[start:end] = start_position
        start = end

        for path_idx in range(n_paths):
            # Create timeline
            duration = int(
                np.random.uniform(
                    config["data"]["vmic"]["movementduration"][0],
                    config["data"]["vmic"]["movementduration"][1],
                )
                * config["data"]["samplerate_position"]
            )
            end += duration

            end_position = randompos(config)

            # Generate trajectories
            coordinates[start:end] = np.linspace(start_position, end_position, duration)

            # Save end coordinates as start for next path
            start_position = end_position
            start = end

            # Add static part at end
            end += (
                int(
                    np.random.uniform(
                        config["data"]["vmic"]["movementpause"][0],
                        config["data"]["vmic"]["movementpause"][1],
                    )
                    * config["data"]["samplerate_position"]
                )
                + 1
            )
            coordinates[start:end] = start_position
            start = end

        # Fill the rest with the last position
        if end < len(timeline):
            coordinates[end:] = start_position

    # Write for data generation
    with open(
        config["data"]["tscdirectory"] + subdir + f"vmic_pos_{idx}.csv",
        mode="w",
        newline="",
    ) as file:
        writer = csv.writer(file)
        for t, coord in zip(timeline, coordinates):
            formatted_row = [f"{t:.2f}"] + [f"{c:.12f}" for c in coord]
            writer.writerow(formatted_row)

    offset_idx = int(config["data"]["offset"] * config["data"]["samplerate_position"])
    timeline_offset = timeline[offset_idx:] - config["data"]["offset"]

    # Write without offset for dataloader
    with open(
        config["data"]["directory"] + subdir + f"vmic_pos_{idx}.csv",
        mode="w",
        newline="",
    ) as file:
        writer = csv.writer(file)
        for t, coord in zip(timeline_offset, coordinates[offset_idx:]):
            formatted_row = [f"{t:.2f}"] + [f"{c:.12f}" for c in coord]
            writer.writerow(formatted_row)

    # %% Export metadata
    metadata = {
        "index": idx,
        "src_n": n_src,
        "src_pos": src_pos,
        "src_az": src_sph[:, 0],
        "src_el": src_sph[:, 1],
        "src_distance": src_sph[:, 2],
        # "src_gain": [src_gain],
        "secsrc_pos": config["data"]["secsrc"]["position"],
        "rmic_pos": np.array(config["data"]["rmic"]["position"])
        * config["data"]["rmic"]["scale"],
        "vmic_movements": n_paths,
        # "vmic_grid_pos": vmic_grid_coords if subdir == "/test/" else None,
        "vmic_startpos": coordinates[0],
    }

    return metadata


# Load configuration
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
n_jobs = int(args.n_jobs)

# Limit number of jobs to available CPUs
if not isinstance(n_jobs, int) or n_jobs < 1:
    n_jobs = os.cpu_count() // 2

n_jobs = min(n_jobs, os.cpu_count())

# Split datasets
N_SCENES_TRAIN = int(config["data"]["nscenes"] * config["data"]["train_val_test"][0])
N_SCENES_VALID = int(config["data"]["nscenes"] * config["data"]["train_val_test"][1])
N_SCENES_TEST = int(config["data"]["nscenes"] * config["data"]["train_val_test"][2])

# Random seed
np.random.seed(config["data"]["seed"])

# Create directories
os.system("rm -rf " + config["data"]["directory"] + "/train")
os.system("rm -rf " + config["data"]["directory"] + "/valid")
os.system("rm -rf " + config["data"]["directory"] + "/test")

os.makedirs(config["data"]["directory"], exist_ok=True)
if N_SCENES_TRAIN > 0:
    os.makedirs(config["data"]["directory"] + "/train", exist_ok=False)
if N_SCENES_VALID > 0:
    os.makedirs(config["data"]["directory"] + "/valid", exist_ok=False)
if N_SCENES_TEST > 0:
    os.makedirs(config["data"]["directory"] + "/test", exist_ok=False)

os.system("rm -rf " + config["data"]["tscdirectory"] + "/train")
os.system("rm -rf " + config["data"]["tscdirectory"] + "/valid")
os.system("rm -rf " + config["data"]["tscdirectory"] + "/test")

os.makedirs(config["data"]["tscdirectory"], exist_ok=True)
if N_SCENES_TRAIN > 0:
    os.makedirs(config["data"]["tscdirectory"] + "/train", exist_ok=False)
if N_SCENES_VALID > 0:
    os.makedirs(config["data"]["tscdirectory"] + "/valid", exist_ok=False)
if N_SCENES_TEST > 0:
    os.makedirs(config["data"]["tscdirectory"] + "/test", exist_ok=False)

# Generate training projects
if N_SCENES_TRAIN > 0:
    metadata = Parallel(n_jobs=n_jobs)(
        delayed(generate_tascar_project)(config, "/train/", idx)
        for idx in tqdm(range(N_SCENES_TRAIN), leave=False)
    )
    ordered_metadata = sorted(metadata, key=lambda x: x["index"])
    with open(config["data"]["tscdirectory"] + "/train/metadata.pkl", "wb") as f:
        pickle.dump(ordered_metadata, f)

# Generate validation projects
if N_SCENES_VALID > 0:
    metadata = Parallel(n_jobs=n_jobs)(
        delayed(generate_tascar_project)(config, "/valid/", idx)
        for idx in tqdm(range(N_SCENES_VALID), leave=False)
    )
    ordered_metadata = sorted(metadata, key=lambda x: x["index"])
    with open(config["data"]["tscdirectory"] + "/valid/metadata.pkl", "wb") as f:
        pickle.dump(ordered_metadata, f)

# Generate test projects
if N_SCENES_TEST > 0:
    metadata = Parallel(n_jobs=n_jobs)(
        delayed(generate_tascar_project)(config, "/test/", idx)
        for idx in tqdm(range(N_SCENES_TEST), leave=False)
    )
    ordered_metadata = sorted(metadata, key=lambda x: x["index"])
    with open(config["data"]["tscdirectory"] + "/test/metadata.pkl", "wb") as f:
        pickle.dump(ordered_metadata, f)
