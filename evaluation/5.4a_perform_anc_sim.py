# Template for ANC co-simulation in python
#
# Parts of the code are reused from the following article:
# F. Holzmüller, C. Blöcher, and A. Sontacchi, “Assessment of simulations in FAUST and TASCAR for the
# development of audio algorithms in acoustic environments,” in Proceedings of the International Faust Conference (IFC-24), Soundmit, Turin, Italy, Nov. 2024.
# https://zenodo.org/records/13859827

import numpy as np
from scipy import signal
import soundfile as sf
from tqdm import tqdm
import toml
from joblib import Parallel, delayed
import pandas as pd
import os

# Parameters
config = toml.load("configs/mod_D6_S3_L32_II32_Ctcn512.toml")
fs_anc = config["samplerate"]
rmt_delay = config["model"]["delay"]
start_anc = 1 * fs_anc
analysis_start = 16 * fs_anc

nfft = 1024

sec_order = 64
control_order = 256

alpha = 1e-1
beta = 1

dataset_dir = "./data/test_static_singlesource/test/"
d_dir = "./rendered_testsets/" + config["filename"] + "/test_static_singlesource/"


def anc_simulation(scene_idx: int):
    # Load data
    x, fs = sf.read(dataset_dir + "innovation_signals/" + f"innov_{scene_idx}.wav")
    de_hat_del, fs = sf.read(d_dir + f"{scene_idx}_rendered.wav")
    de, fs = sf.read(dataset_dir + f"scene_{scene_idx}.wav")
    dm = de[..., 0:-1]
    de = de[..., -1]

    data_length = np.min([de_hat_del.shape[0], de.shape[0]])

    x = x[0:data_length]
    de_hat_del = de_hat_del[0:data_length]
    de = de[0:data_length]
    dm = dm[0:data_length, :]

    # # Prepend zeros to de_hat to account for already delayed estimate
    de_hat = np.concatenate((np.zeros(config["hopsize"]), de_hat_del))

    # Load secondary path ir
    sec_path, fs = sf.read(dataset_dir + f"ir_{scene_idx}.wav")
    g_e = sec_path[0:sec_order, -1]
    g_m = sec_path[0:sec_order, 0:-1]

    # Calculate filtered reference with predelay
    x_f = signal.convolve(x, g_e, mode="full")

    x_fdm = np.zeros((len(x_f), dm.shape[1]))

    for ch in range(dm.shape[1]):
        x_fdm[..., ch] = signal.convolve(x, g_m[..., ch], mode="full")

    x_f = x_f[0:data_length]
    x_fdm = x_fdm[0:data_length, :]

    # Adjust reference signal lengths and initialize control-, error signals, and coefficients
    u_target = np.zeros_like(de)
    u_obstasnet = np.zeros_like(de)
    u_mpanc = np.zeros_like(de)

    ye_target = np.zeros_like(de)
    ye_obstasnet = np.zeros_like(de)
    ye_mpanc = np.zeros_like(de)
    ye_hat = np.zeros_like(de)

    e_target = np.zeros_like(de)
    e_obstasnet = np.zeros_like(de)
    e_mpanc = np.zeros_like(de)
    e_hat = np.zeros_like(de)

    m = np.zeros_like(dm)
    ym = np.zeros_like(dm)

    # Initialize filter coefficients
    coeffs_target = np.zeros((control_order,))
    coeffs_obstasnet = np.zeros((control_order,))
    coeffs_mpanc = np.zeros((control_order,))

    # Run without adaptation (equivalent to e = d, since coeffs are 0)
    e_target[:start_anc] = de[:start_anc]
    e_obstasnet[:start_anc] = de[:start_anc]
    e_mpanc[:start_anc] = de[:start_anc]
    e_hat[: start_anc - rmt_delay] = de_hat[: start_anc - rmt_delay]
    m[:start_anc, :] = dm[:start_anc, :]

    # Run ANC from designated start time
    for i in range(start_anc, data_length):
        u_target[i] = np.dot(x[i : i - control_order : -1], coeffs_target)
        u_obstasnet[i] = np.dot(x[i : i - control_order : -1], coeffs_obstasnet)
        u_mpanc[i] = np.dot(
            x[i : i - control_order : -1], coeffs_mpanc
        )  # Reference signal for multi-channel ANC

        ye_target[i] = np.dot(u_target[i : i - len(g_e) : -1], g_e)
        ye_obstasnet[i] = np.dot(u_obstasnet[i : i - len(g_e) : -1], g_e)
        ye_mpanc[i] = np.dot(u_mpanc[i : i - len(g_e) : -1], g_e)

        ym[i] = np.dot(u_mpanc[i : i - len(g_m) : -1], g_m)

        e_target[i] = de[i] + ye_target[i]
        e_obstasnet[i] = de[i] + ye_obstasnet[i]
        e_mpanc[i] = de[i] + ye_mpanc[i]

        m[i, :] = dm[i, :] + ym[i, :]

        ye_hat[i - rmt_delay] = np.dot(
            u_obstasnet[i - rmt_delay : i - rmt_delay - len(g_e) : -1],
            g_e,
        )
        e_hat[i - rmt_delay] = de_hat[i - rmt_delay] + ye_hat[i - rmt_delay]

        mu = alpha / (beta + (1 - beta) * x_f[i - rmt_delay] ** 2)
        mu_mpanc = alpha / (beta + (1 - beta) * x_fdm[i - rmt_delay] ** 2)

        # Update coefficients
        coeffs_target = (
            coeffs_target
            - mu
            * e_target[i - rmt_delay]
            * x_f[i - rmt_delay : i - rmt_delay - control_order : -1]
        )
        coeffs_obstasnet = (
            coeffs_obstasnet
            - mu
            * e_hat[i - rmt_delay]
            * x_f[i - rmt_delay : i - rmt_delay - control_order : -1]
        )
        coeffs_mpanc = coeffs_mpanc - np.mean(
            mu_mpanc
            * m[i - rmt_delay, :]
            * x_fdm[i - rmt_delay : i - rmt_delay - control_order : -1],
            axis=1,
        )

    rms_de = np.sqrt(np.mean(de[analysis_start:] ** 2))
    rms_e_target = np.sqrt(np.mean(e_target[analysis_start:] ** 2))
    rms_e_obstasnet = np.sqrt(np.mean(e_obstasnet[analysis_start:] ** 2))
    rms_e_mpanc = np.sqrt(np.mean(e_mpanc[analysis_start:] ** 2))

    f_axis, PSD_de = signal.welch(de[analysis_start:], nperseg=nfft, fs=fs_anc)
    _, PSD_e_target = signal.welch(e_target[analysis_start:], nperseg=nfft, fs=fs_anc)
    _, PSD_e_obstasnet = signal.welch(
        e_obstasnet[analysis_start:], nperseg=nfft, fs=fs_anc
    )
    _, PSD_e_mpanc = signal.welch(e_mpanc[analysis_start:], nperseg=nfft, fs=fs_anc)

    nr_target = 20 * np.log10(rms_e_target / rms_de)
    nr_obstasnet = 20 * np.log10(rms_e_obstasnet / rms_de)
    nr_mpanc = 20 * np.log10(rms_e_mpanc / rms_de)

    NR_target = 10 * np.log10(PSD_de / PSD_e_target)
    NR_obstasnet = 10 * np.log10(PSD_de / PSD_e_obstasnet)
    NR_mpanc = 10 * np.log10(PSD_de / PSD_e_mpanc)

    return {
        "NR_target": NR_target,
        "NR_obstasnet": NR_obstasnet,
        "NR_mpanc": NR_mpanc,
        "f_axis": f_axis,
        "nr_bb_target": nr_target,
        "nr_bb_obstasnet": nr_obstasnet,
        "nr_bb_mpanc": nr_mpanc,
    }


if __name__ == "__main__":
    n_scenes = 1000
    n_jobs = os.cpu_count()

    results = Parallel(n_jobs=n_jobs)(
        delayed(anc_simulation)(scene_idx) for scene_idx in tqdm(range(n_scenes))
    )

    # Merge results into DataFrame
    results_df = pd.DataFrame(results)
    results_df.to_pickle(f"export/anc_sim_results_{config['filename']}.pkl")
