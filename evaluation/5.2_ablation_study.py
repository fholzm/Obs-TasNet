import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams["backend"] = "Agg"  # Use Agg backend for non-interactive plotting

models = ["mod_D6_S3_L32_II32", "orig_D6_S3_L32_II32"]
model_labels = ["Obs-TasNet", "Obs-TasNet (w/o $L$-BN)"]
datasets = ["static", "dynamic"]

linestyles = ["-", "-."]
color_model = ["tab:blue", "tab:red"]

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


for idx_ds, dataset in enumerate(datasets):
    plt.figure(figsize=(fig_width / 72, fig_width / 1.8 / 72))

    for idx_model, model in enumerate(models):
        fn = f"export/{model}_test_{dataset}_singlesource_epoch_49.npz"

        data = np.load(fn, allow_pickle=True)

        print(
            f"Mean NMSE for {dataset} dataset and {model}: {np.mean(10 * np.log10(data['test_NMSE_per_epoch_samplewise'])):.2f} dB"
        )

        est_error = np.mean(10 * np.log10(data["est_error_psd"]), axis=0)
        f_axis = np.linspace(0, fs / 2, est_error.shape[0])

        plt.semilogx(
            f_axis,
            est_error,
            color=color_model[idx_model],
            linestyle=linestyles[idx_model],
            label=model_labels[idx_model],
        )

    plt.legend(loc="lower right")
    plt.grid(which="both")

    plt.xlabel("Frequency / Hz")
    plt.ylabel("$E$ / dB")

    plt.ylim([-48, 2])

    plt.tight_layout(pad=0.1)

    plt.savefig(f"figures/ablation_study_est_error_{dataset}.pdf", bbox_inches="tight")
    plt.savefig(f"figures/ablation_study_est_error_{dataset}.eps", bbox_inches="tight")
