import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.rcParams["backend"] = "Agg"  # Use Agg backend for non-interactive plotting


fn = "export/anc_sim_results_mod_D6_S3_L32_II32_Ctcn512.pkl"
results_df = pd.read_pickle(fn)
filename = fn.split("/")[-1].split(".pkl")[0].replace("anc_sim_results_", "")

fig_width = 230  # in pt
plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["CMU Serif"],
        "font.size": 9,  # default text size
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "lines.linewidth": 1.5,
        "axes.linewidth": 0.8,
    }
)

# Remove entries, where NR could not be computed due to divergence
valid_idx = np.where(
    np.isfinite(results_df["nr_bb_target"]) & (results_df["nr_bb_target"] < 0)
)
results_df = results_df.iloc[valid_idx]


# Calculate mean noise reduction
mean_NR_target = np.mean(np.array(results_df["NR_target"].tolist()), axis=0)
mean_NR_obstasnet = np.mean(np.array(results_df["NR_obstasnet"].tolist()), axis=0)
mean_NR_mpanc = np.mean(np.array(results_df["NR_mpanc"].tolist()), axis=0)
f_axis = results_df.iloc[0]["f_axis"]

mean_nr_bb_target = np.mean(results_df["nr_bb_target"])
mean_nr_bb_obstasnet = np.mean(results_df["nr_bb_obstasnet"])
mean_nr_bb_mpanc = np.mean(results_df["nr_bb_mpanc"])

plt.figure(figsize=(fig_width / 72, fig_width / 2 / 72))
plt.semilogx(f_axis, mean_NR_obstasnet, label="Obs-TasNet")
plt.semilogx(f_axis, mean_NR_mpanc, "-.", label="Multi-point ANC")
plt.grid(which="both")
plt.legend()

plt.gca().invert_yaxis()
plt.xlabel("Frequency / Hz")
plt.ylabel("NR / dB")

plt.tight_layout(pad=0.1)

plt.savefig(f"figures/anc_sim_NR_{filename}.pdf", bbox_inches="tight")
plt.savefig(f"figures/anc_sim_NR_{filename}.eps", bbox_inches="tight")
