import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams["backend"] = "Agg"  # Use Agg backend for non-interactive plotting

fs = 16000

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


# Define model and dataset
model = "mod_D6_S3_L32_II32_Ctcn512"
dataset = "static"

# Load data
fn = f"export/{model}_test_{dataset}_singlesource_epoch_49.npz"
data = np.load(fn, allow_pickle=True)

NMSE = 10 * np.log10(data["test_NMSE_per_epoch_samplewise"])
est_error_psd = 10 * np.log10(data["est_error_psd"])
vmic_position = np.array(data["metadata"].item()["vmic_startpos"])

radius_borders = np.linspace(0, 5, 6) / 100

mean_NMSE_per_radius = np.zeros(radius_borders.shape[0] - 1)

linestyles = ["-", "--", (5, (10, 3)), "-.", (0, (1, 1))]

f_axis = np.linspace(0, fs / 2, est_error_psd.shape[1])
plt.figure(figsize=(fig_width / 72, fig_width / 1.45 / 72))

for idx_radius in range(radius_borders.shape[0] - 1):
    idx_samples = np.where(
        (np.linalg.norm(vmic_position, axis=1) >= radius_borders[idx_radius])
        & (np.linalg.norm(vmic_position, axis=1) < radius_borders[idx_radius + 1])
    )[0]

    mean_NMSE_per_radius[idx_radius] = np.mean(NMSE[idx_samples])

    plt.semilogx(
        f_axis,
        est_error_psd[idx_samples].mean(axis=0),
        linestyle=linestyles[idx_radius % len(linestyles)],
        label=f"\(a \in [{radius_borders[idx_radius]*100:.0f}; {radius_borders[idx_radius + 1]*100:.0f})\) cm",
    )

plt.legend(
    loc="lower center",
    bbox_to_anchor=(0.5, 1.05),
    ncol=2,
    borderaxespad=0,
    frameon=True,
)

plt.grid(which="both")
plt.xlabel("Frequency / Hz")
plt.ylabel("$E$ / dB")
plt.ylim([-52, 2])

plt.tight_layout(pad=0.1)

plt.savefig(f"figures/radial_error.pdf", bbox_inches="tight")
plt.savefig(f"figures/radial_error.eps", bbox_inches="tight")

print("Mean NMSE per radius:", mean_NMSE_per_radius)
