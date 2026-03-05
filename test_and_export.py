import os
import toml
import torch, torchaudio
import io
import numpy as np
import sys
import tqdm
import matplotlib
import matplotlib.pyplot as plt
import argparse
from typing import Union
import PIL.Image
import wandb
import signal
from collections import defaultdict
import itertools

from torch.utils import data
from ptflops import get_model_complexity_info
from utils.dataset import DirectionalNoiseDatasetPrerendered, custom_collate_fn
from utils import models
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import ToTensor
from utils.metrics import NMSE
from utils.transforms import OverlapSave


def cleanup_and_exit(signum, frame):
    wandb.finish()
    print("Caught signal:", signum)
    print("Releasing GPU resources...")
    torch.cuda.empty_cache()  # Free unused GPU memory
    sys.exit(0)  # Exit the script gracefully


def load_checkpoint(dirname: str, file_list: str, extension: str):
    # get latest checkpoint
    epochs = [i.split("_", -1)[-1] for i in file_list]
    epochs = [int(i.split(".", -1)[0]) for i in epochs]
    latest_epoch = 49  # Load model after 50 epochs trianing
    latest_substring = "_" + str(latest_epoch) + extension
    latest_ckpts = [latest_substring in d for d in file_list]
    temp = np.array(file_list)
    latest_ckpt_files = temp[latest_ckpts]

    try:
        assert len(latest_ckpt_files) == 3
    except AssertionError:
        sys.exit(
            "there exist either too many checkpoint-files or one checkpoint-file is missing!"
        )

    model_idx = np.array(["model" in f for f in latest_ckpt_files])
    latest_model_ckpt = latest_ckpt_files[model_idx][0]

    model_state_dict = torch.load(
        os.path.join(dirname, latest_model_ckpt), map_location="cpu"
    )

    print(
        "checkpoints '"
        + latest_model_ckpt
        + "' for test_epoch "
        + str(latest_epoch)
        + " has been loaded!"
    )
    return model_state_dict, latest_epoch


def plot_spec_singlebatch(
    psd: np.array,
    f_axis: np.array,
    vmin: float = None,
    vmax: float = None,
    log_faxis: bool = True,
    dB: bool = True,
    title: str = None,
):
    nBins = psd.shape[0]

    if dB:
        psd[psd < 1e-10] = 1e-10  # Avoid log(0)
        psd[np.isnan(psd)] = 1e10
        psd = 10 * np.log10(np.real(np.abs(psd)))

    if f_axis is None:
        f_axis = np.arange(0, nBins)
        f_axis_label = "Frequency"
    else:
        f_axis_label = "Frequency / Hz"

    if vmin is None:
        vmin = np.min(psd) * 0.9
    if vmax is None:
        vmax = np.max(psd) * 1.1

    plt.figure()

    if log_faxis:
        plt.semilogx(f_axis, psd)
    else:
        plt.plot(f_axis, psd)

    plt.ylim([vmin, vmax])
    plt.grid(which="both")
    plt.xlabel(f_axis_label)

    if dB:
        plt.ylabel("Relative error / dB")
    else:
        plt.ylabel("Relative error")

    if title is not None:
        plt.title(title)
    plt.axis("auto")

    buf = io.BytesIO()
    plt.savefig(buf, format="jpeg")
    plt.close()
    buf.seek(0)

    return buf


def write_loss_summary(
    tag: str, tb_writer: SummaryWriter, wandb_run, loss: float, step: int
):
    tb_writer.add_scalar(tag, loss, step)
    wandb_run.log({tag: loss}, step=step)
    tb_writer.close()


def write_losshist_summary(
    tag: str, tb_writer: SummaryWriter, wandb_run, loss_hist: np.array, step: int
):
    # Remove entries with -inf or inf or NaN
    loss_hist = loss_hist[np.isfinite(loss_hist)]
    tb_writer.add_histogram(tag, loss_hist, step)
    wandb_run.log({tag: wandb.Histogram(loss_hist)}, step=step)
    tb_writer.close()


def write_pyplotfigure_summary(
    tag: str, tb_writer: SummaryWriter, wandb_run, plot_buf: io.BytesIO, step: int
):
    image = PIL.Image.open(plot_buf)
    wandb_run.log({tag: wandb.Image(image)}, step=step)

    image = ToTensor()(image)

    tb_writer.add_image(tag, image, step)
    tb_writer.close()


# %% Define test cases
testsets = [
    "test_static_singlesource",
    "test_dynamic_singlesource",
]
testset_path = "./data/"

output_dir = "./rendered_testsets/"


# %% Setup signal handlers
signal.signal(signal.SIGINT, cleanup_and_exit)  # Handle Ctrl+C
signal.signal(signal.SIGTERM, cleanup_and_exit)  # Handle termination signals

matplotlib.use("Agg")  # Use non-interactive backend for tensorboard logging
# torch.autograd.set_detect_anomaly(True)
# %% Get config file provided as argument
parser = argparse.ArgumentParser()
parser.add_argument(
    "-c",
    "--config",
    nargs="?",
    const=1,
    type=str,
    default="configs/mod_D6_S3_L32_II32_Ctcn512.toml",
)

args = parser.parse_args()
config = toml.load(args.config)

os.environ["CUDA_VISIBLE_DEVICES"] = str(config["cuda_visible_devices"])

# %% Create directories for rendered testsignals
output_dir = os.path.join(output_dir, config["filename"])
os.makedirs(output_dir, exist_ok=True)


# %% Load heavily used config parameters
NFFT = config["nfft"]
HOPSIZE = config["hopsize"]
FS = config["samplerate"]
EPOCHS = config["train"]["max_epochs"]
CKPT_PATH = os.path.join(config["checkpoint_path"], config["filename"])
INFERENCE_INTERVAL = config["valid"]["inference_interval"]
BATCHSIZE = 50

# Parameters for encoder
C = (
    len(config["data"]["rmic"]["position"]) + 1
)  # Number of physical microphones plus encoded position
L = config["model"]["L"]  # Number of overlapping input segments
F = config["model"]["F"]  # Number of output features of encoder

assert L >= INFERENCE_INTERVAL, "L must be equal or larger than inference interval!"

# Parameters for bottleneck
F_b = config["model"]["F_b"]  # Number of kernels in bottleneck
C_b = config["model"]["C_b"]  # Number of channels in TCN
C_TCN = config["model"]["C_TCN"]  # Number of hidden layers in TCN
D = config["model"]["D"]  # Number of dilated layers in TCN
TCN_kernelsize = config["model"]["TCN_kernelsize"]  # Kernel size of TCN
S = config["model"]["S"]  # Number of stacks in TCN

# Calculate derived parameters
N_REM_MICS = len(config["data"]["rmic"]["position"])
aperture = 2 * np.sqrt(2) * config["data"]["rmic"]["scale"]
max_lag = int(np.ceil(np.max(aperture) / config["data"]["c"] * FS))
n_coeffs = config["nfft"] - config["hopsize"] + 1

# %% Set random seed and devices
np.random.seed(config["train"]["seed"])

if torch.cuda.is_available():
    test_device = torch.device("cuda")
    torch.cuda.empty_cache()
    print("CUDA is available!")

elif torch.backends.mps.is_available():
    test_device = torch.device("mps")
    print("MPS is available!")
else:
    test_device = torch.device("cpu")
    print("CUDA and MPS are not available!")


# %% Initialize model, optimizer and loss
model_to_train = getattr(models, config["model"]["name"])(config)
model_to_train.to(test_device)
model_to_train.eval()

input_fun = lambda inp_shape: {
    "x": torch.FloatTensor(torch.empty(inp_shape)).to(test_device),
    "virt_coords": torch.FloatTensor(torch.empty(1, 3, L)).to(test_device),
}

macs, params = get_model_complexity_info(
    model_to_train,
    (1, N_REM_MICS, NFFT + HOPSIZE * (L - 1)),
    input_constructor=input_fun,
    as_strings=True,
    backend="pytorch",
    print_per_layer_stat=True,
    verbose=True,
)

print("{:<30}  {:<8}".format("Computational complexity: ", macs))
print("{:<30}  {:<8}".format("Number of parameters: ", params))

loss_fn = torch.nn.MSELoss()
NMSE = NMSE(return_dB=False, per_sample=True)

# %% Load checkpoint if available
ckpt_file_list = os.listdir(CKPT_PATH)
if len(ckpt_file_list) >= 3:
    # if at least two checkpoint files (model and optimizer) exist => load checkpoints
    latest_model_ckpt, test_epoch = load_checkpoint(CKPT_PATH, ckpt_file_list, ".ckpt")
    model_to_train.load_state_dict(latest_model_ckpt)
else:
    RuntimeError("No checkpoint found!")
    # exit script
    sys.exit(1)

for testset in tqdm.tqdm(testsets):
    # %% Setup tensorboard and wandb
    writer = SummaryWriter(
        os.path.join(config["tensorboard_path"], config["filename"] + f"_{testset}"),
        filename_suffix=".tlog",
    )

    # wandb.login()
    run = wandb.init(
        project="DeepObservationFilter_test",
        name=config["filename"] + f"_{testset}",
        config=config,
        # mode="online" if config["log_wandb"] else "disabled",
        mode="disabled",
    )

    current_testset_path = os.path.join(testset_path, testset)
    output_path = os.path.join(output_dir, testset)
    os.makedirs(output_path, exist_ok=False)

    test_ds = DirectionalNoiseDatasetPrerendered(
        config, current_testset_path + "/test/"
    )

    test_dl = data.DataLoader(
        test_ds,
        batch_size=BATCHSIZE,
        num_workers=config["valid"]["num_process"],
        shuffle=False,
        drop_last=True,
        collate_fn=custom_collate_fn,
    )

    N_BATCHES = (
        len(test_dl)
        if config["valid"]["n_batches"] == -1
        else config["valid"]["n_batches"]
    )

    # %% Allocate lists for metrics
    OS = OverlapSave(nfft=NFFT, hopsize=HOPSIZE, complex_input=False)

    with torch.no_grad():
        test_loss_per_batch = []

        torch.manual_seed(
            config["valid"]["seed"] + 42
        )  # Different seed than validation

        # Initialize lists for samplewise metrics
        metadata_samplewise = []
        test_NMSE_per_epoch_samplewise = torch.zeros((N_BATCHES, BATCHSIZE))
        est_error_psd = np.zeros(
            (
                N_BATCHES,
                BATCHSIZE,
                config["valid"]["psd_nfft"] // 2 + 1,
            ),
        )

        model_to_train.eval()

        for batch, (rm, vm, metadata, vmic_pos) in enumerate(
            tqdm.tqdm(
                test_dl,
                total=N_BATCHES,
                leave=False,
                desc="validation batches",
                miniters=int(N_BATCHES / 20),
                disable=True,
            )
        ):
            if batch >= N_BATCHES:
                break
            metadata_samplewise.append(metadata)

            rm = rm.to(test_device)
            vm = vm.to(test_device)
            vmic_pos = vmic_pos.to(test_device)

            # Estimate number of signal blocks
            n_blocks = int(np.floor(rm.shape[-1] / HOPSIZE)) - 1
            n_steps = int(np.floor(n_blocks / INFERENCE_INTERVAL))

            # Convert time to samples for postition data
            vmic_pos[:, 0, :] *= FS
            rm_fd = OS(rm, to_fd=True, reset=True)

            vm_est_fd_batch = torch.zeros(
                (rm_fd.shape[0], rm_fd.shape[2], rm_fd.shape[3]),
                dtype=torch.complex64,
                requires_grad=False,
                device=test_device,
            )

            coeffs = torch.zeros(
                (rm.shape[0], C - 1, NFFT // 2 + 1),
                device=rm.device,
                dtype=torch.complex64,
            )

            # Find start segment index, so that no negative blocks are used
            segment_start = np.ceil(L / INFERENCE_INTERVAL).astype(int) - 1

            for segment in range(segment_start, n_steps):
                start_segment_OA = segment * INFERENCE_INTERVAL
                end_segment = (segment + 1) * INFERENCE_INTERVAL
                start_segment_inf = end_segment - L

                start_sample_inf = (start_segment_inf) * HOPSIZE
                start_sample_vm = (start_segment_OA + 1) * HOPSIZE
                end_sample = (end_segment - 1) * HOPSIZE + NFFT

                if end_sample > vm.shape[2]:
                    end_sample = vm.shape[2]

                # Define required time segments to query virtual mic position
                # --> corresponding to last sample in each segment
                t_segments_vmic = (
                    torch.arange(start_segment_inf + 1, end_segment + 1).to(test_device)
                    * NFFT
                    - 1
                )

                # Compensate delay on vm position
                t_segments_vmic += config["model"]["delay"]

                # Find closest position data for each time segment
                virtual_pos = torch.zeros((vmic_pos.shape[0], 3, L), device=test_device)
                for idx, t in enumerate(t_segments_vmic):
                    delta_t = vmic_pos[0, 0, :] - t
                    # take the highest negative delta_t --> the last position before t
                    t_idx = torch.argmax(delta_t[delta_t < 0])

                    virtual_pos[:, :, idx] = vmic_pos[:, 1:, t_idx]

                # Perform filter operation
                vm_est_fd = torch.sum(
                    rm_fd[..., start_segment_OA:end_segment] * coeffs.unsqueeze(-1),
                    1,
                )
                vm_est_fd_batch[..., start_segment_OA:end_segment] = vm_est_fd

                vm_estimated = OS(
                    vm_est_fd,
                    to_fd=False,
                )

                # Calculate loss & backprop only after first iteration
                vm_target = torch.squeeze(vm[..., start_sample_vm:end_sample])

                if vm_target.shape[-1] > vm_estimated.shape[-1]:
                    vm_target = vm_target[..., : vm_estimated.shape[-1]]
                elif vm_target.shape[-1] < vm_estimated.shape[-1]:
                    vm_estimated = vm_estimated[..., : vm_target.shape[-1]]

                if config["train"]["normalized_loss"]:
                    rms = torch.sqrt(torch.mean(vm_target**2, dim=-1)).unsqueeze(-1)
                    loss = loss_fn(vm_estimated / rms, vm_target / rms)
                else:
                    loss = loss_fn(vm_estimated, vm_target)

                test_loss_per_batch.append(loss.detach().cpu().numpy())

                # Calculate coefficients for use in next segment
                if segment < n_steps - 1:
                    coeffs = model_to_train(
                        rm[..., start_sample_inf:end_sample],
                        virtual_pos,
                    )

                    if torch.isnan(coeffs).any() or torch.isinf(coeffs).any():
                        print(
                            "NaN, inf or -inf found in coefficients! Stopping validation at test_epoch "
                            + str(test_epoch)
                        )

            start_sample = L * HOPSIZE
            start_sample_vm = start_sample + HOPSIZE

            vm_estimated = OS(vm_est_fd_batch[..., L:], to_fd=False)
            vm_estimated_full = OS(vm_est_fd_batch[..., 0:], to_fd=False)

            vm_target = torch.squeeze(vm[..., start_sample_vm:])

            max_vm_length = torch.min(
                torch.tensor([vm_target.shape[-1], vm_estimated.shape[-1]])
            )

            vm_target = vm_target[..., :max_vm_length]
            vm_estimated = vm_estimated[..., :max_vm_length]
            vm_estimated_full = vm_estimated_full[..., : max_vm_length + L * HOPSIZE]

            # Calculate NMSE for each sample in batch
            test_NMSE_per_epoch_samplewise[batch, :] = NMSE(vm_target, vm_estimated)

            # Calculate PSD using torchaudio
            win_psd = torch.hann_window(config["valid"]["psd_nfft"]).to(test_device)
            vm_target_stft_for_psd = torch.unsqueeze(
                torch.stft(
                    vm_target,
                    n_fft=config["valid"]["psd_nfft"],
                    hop_length=config["valid"]["psd_nfft"] // 2,
                    window=win_psd,
                    onesided=True,
                    return_complex=True,
                ),
                1,
            )
            difference_stft_for_psd = torch.unsqueeze(
                torch.stft(
                    vm_target - vm_estimated,
                    n_fft=config["valid"]["psd_nfft"],
                    hop_length=config["valid"]["psd_nfft"] // 2,
                    window=win_psd,
                    onesided=True,
                    return_complex=True,
                ),
                1,
            )

            f_axis = np.linspace(0, FS // 2, config["valid"]["psd_nfft"] // 2 + 1)
            est_error_diff = torchaudio.functional.psd(difference_stft_for_psd)[
                ..., 0, 0
            ].real
            psd_target = torchaudio.functional.psd(vm_target_stft_for_psd)[
                ..., 0, 0
            ].real

            est_error_psd[batch] = (est_error_diff / psd_target).cpu().numpy()

            # Save rendered audio files for each sample in batch
            for ii in range(vm_estimated_full.shape[0]):
                fn = os.path.join(
                    output_path,
                    str(metadata_samplewise[-1][ii]["index"]) + "_rendered.wav",
                )
                torchaudio.save(
                    fn,
                    vm_estimated_full[ii, :].unsqueeze(0).cpu(),
                    FS,
                    encoding="PCM_S",
                    bits_per_sample=32,
                )

        # Save metrics for each sample in batch
        fn = os.path.join(
            config["valid"]["export_path"],
            config["filename"] + f"_{testset}_epoch_{test_epoch}.npz",
        )

        # Convert metadata dict content to numpy array
        tmp_dict = {}

        agg = defaultdict(list)
        for d in itertools.chain.from_iterable(
            metadata_samplewise
        ):  # flatten one level
            for k, v in d.items():
                agg[k].append(v)

        metadata_agg = dict(agg)

        np.savez(
            fn,
            est_error_psd=est_error_psd.reshape(
                est_error_psd.shape[0] * est_error_psd.shape[1], -1
            ),
            test_NMSE_per_epoch_samplewise=test_NMSE_per_epoch_samplewise.flatten()
            .cpu()
            .numpy(),
            metadata=metadata_agg,
        )

        test_NMSE_per_epoch_samplewise[test_NMSE_per_epoch_samplewise < 1e-10] = (
            1e-10  # Avoid log(0)
        )
        test_NMSE_per_epoch_samplewise[torch.isnan(test_NMSE_per_epoch_samplewise)] = (
            1e10
        )
        test_NMSE_per_epoch_samplewise = 10 * torch.log10(
            test_NMSE_per_epoch_samplewise
        )

        # Calculate mean and median NMSE for each sample in batch and show in tensorboard/wandb
        test_NMSE_per_epoch = torch.mean(test_NMSE_per_epoch_samplewise).cpu().numpy()
        test_best_NMSE_per_epoch = (
            torch.min(test_NMSE_per_epoch_samplewise).cpu().numpy()
        )
        test_worst_NMSE_per_epoch = (
            torch.max(test_NMSE_per_epoch_samplewise).cpu().numpy()
        )
        test_median_NMSE_per_epoch = (
            torch.median(test_NMSE_per_epoch_samplewise).cpu().numpy()
        )
        test_mean_error_fd_per_epoch = np.mean(est_error_psd, axis=(0, 1))
        test_median_error_fd_per_epoch = np.median(est_error_psd, axis=(0, 1))

        im_to_plot = plot_spec_singlebatch(
            test_mean_error_fd_per_epoch,
            f_axis=f_axis,
            vmin=-30,
            vmax=15,
            log_faxis=True,
            dB=True,
            title="mean validation error",
        )
        write_pyplotfigure_summary(
            "mean test error", writer, run, im_to_plot, test_epoch
        )

        im_to_plot = plot_spec_singlebatch(
            test_median_error_fd_per_epoch,
            f_axis=f_axis,
            vmin=-30,
            vmax=15,
            log_faxis=True,
            dB=True,
            title="median test error",
        )
        write_pyplotfigure_summary(
            "median test error", writer, run, im_to_plot, test_epoch
        )

        test_loss_per_epoch = np.mean(test_loss_per_batch)

        write_loss_summary(
            "test loss (MSE)", writer, run, test_loss_per_epoch, test_epoch
        )

        write_loss_summary(
            "mean test NMSE in dB",
            writer,
            run,
            test_NMSE_per_epoch,
            test_epoch,
        )
        write_loss_summary(
            "median test NMSE in dB",
            writer,
            run,
            test_median_NMSE_per_epoch,
            test_epoch,
        )
        write_loss_summary(
            "minimum test NMSE in dB",
            writer,
            run,
            test_worst_NMSE_per_epoch,
            test_epoch,
        )
        write_loss_summary(
            "maximum test NMSE in dB",
            writer,
            run,
            test_best_NMSE_per_epoch,
            test_epoch,
        )
        write_losshist_summary(
            "test NMSE in dB",
            writer,
            run,
            test_NMSE_per_epoch_samplewise.cpu().numpy(),
            test_epoch,
        )

        wandb.finish()
