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


def write_audio_summary(
    tag: str,
    tb_writer: SummaryWriter,
    wandb_run,
    vm_target: torch.Tensor,
    vm_estimated: torch.Tensor,
    error: torch.Tensor,
    step: int,
    fs: int,
):

    full_tag_target = tag + "/target"
    full_tag_est = tag + "/estimated"
    full_tag_error = tag + "/error"

    tb_writer.add_audio(full_tag_target, vm_target, step, sample_rate=fs)
    tb_writer.add_audio(full_tag_est, vm_estimated, step, sample_rate=fs)
    tb_writer.add_audio(full_tag_error, error, step, sample_rate=fs)
    tb_writer.close()

    wandb_run.log(
        {
            full_tag_target: wandb.Audio(vm_target, sample_rate=fs),
            full_tag_est: wandb.Audio(vm_estimated, sample_rate=fs),
            full_tag_error: wandb.Audio(error, sample_rate=fs),
        },
        step=step,
    )


def write_checkpoint(
    obj: Union[
        torch.optim.Optimizer, torch.nn.Module, torch.optim.lr_scheduler.ExponentialLR
    ],
    name: str,
    dir: str,
    epoch: int,
    extension="ckpt",
):

    filename = name + str(epoch) + "." + extension
    cp_name = os.path.join(dir, filename)
    torch.save(obj.state_dict(), cp_name)
    print("Checkpoint '" + filename + "' for epoch " + str(epoch) + " has been stored.")


def load_checkpoint(dirname: str, file_list: str, extension: str):
    # get latest checkpoint
    epochs = [i.split("_", -1)[-1] for i in file_list]
    epochs = [int(i.split(".", -1)[0]) for i in epochs]
    latest_epoch = max(epochs)
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
    optim_idx = np.array(["optimizer" in f for f in latest_ckpt_files])
    sched_idx = np.array(["scheduler" in f for f in latest_ckpt_files])
    latest_model_ckpt = latest_ckpt_files[model_idx][0]
    latest_opt_ckpt = latest_ckpt_files[optim_idx][0]
    latest_sched_ckpt = latest_ckpt_files[sched_idx][0]

    model_state_dict = torch.load(
        os.path.join(dirname, latest_model_ckpt), map_location="cpu"
    )
    opt_state_dict = torch.load(
        os.path.join(dirname, latest_opt_ckpt), map_location="cpu", weights_only=False
    )
    sched_state_dict = torch.load(
        os.path.join(dirname, latest_sched_ckpt), map_location="cpu", weights_only=False
    )

    print(
        "checkpoints '"
        + latest_model_ckpt
        + "', '"
        + latest_opt_ckpt
        + "', and '"
        + latest_sched_ckpt
        + "' for epoch "
        + str(latest_epoch)
        + " have been loaded!"
    )
    return model_state_dict, opt_state_dict, sched_state_dict, latest_epoch


def plot_spec_tb(
    mtx: np.array,
    epochs: np.array = None,
    f_axis: np.array = None,
    vmin: float = -20,
    vmax: float = 10,
    dB: bool = True,
    title: str = None,
):
    nBins = mtx.shape[0]
    nEpochs = mtx.shape[1]

    if epochs is None:
        epochs = np.arange(0, nEpochs)

    if f_axis is None:
        f_axis = np.arange(0, nBins)
        f_axis_label = "Frequency"
    else:
        f_axis_label = "Frequency / Hz"

    if dB:
        mtx[mtx < 1e-10] = 1e-10  # Avoid log(0)
        mtx[np.isnan(mtx)] = 1e10
        mtx = 10 * np.log10(mtx)

    xmin = epochs[0]
    xmax = epochs[-1]
    extent = xmin, xmax, f_axis[0], f_axis[-1]

    plt.figure()
    plt.imshow(mtx, extent=extent, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
    plt.xlabel("Epochs")
    plt.ylabel(f_axis_label)
    if title is not None:
        plt.title(title)

    cbar = plt.colorbar()
    if dB:
        cbar.set_label("Relative error / dB")
    else:
        cbar.set_label("Relative error")
    plt.axis("auto")

    buf = io.BytesIO()
    plt.savefig(buf, format="jpeg")
    buf.seek(0)

    return buf


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


# Register signal handlers
signal.signal(signal.SIGINT, cleanup_and_exit)  # Handle Ctrl+C
signal.signal(signal.SIGTERM, cleanup_and_exit)  # Handle termination signals

matplotlib.use("Agg")  # Use non-interactive backend for tensorboard logging
# torch.autograd.set_detect_anomaly(True)
# %% Get config file provided as argument
parser = argparse.ArgumentParser()
parser.add_argument(
    "-c", "--config", nargs="?", const=1, type=str, default="config.toml"
)

args = parser.parse_args()
config = toml.load(args.config)

os.environ["CUDA_VISIBLE_DEVICES"] = str(config["cuda_visible_devices"])

# %% Load heaivily used config parameters
NFFT = config["nfft"]
HOPSIZE = config["hopsize"]
FS = config["samplerate"]
EPOCHS = config["train"]["max_epochs"]
VALID_ONLY = config["valid"]["only"]
CKPT_PATH = os.path.join(config["checkpoint_path"], config["filename"])
if config["train"]["pretrained"]:
    CKPT_PATH_LOAD = CKPT_PATH + "_pretrained"
else:
    CKPT_PATH_LOAD = CKPT_PATH

INFERENCE_INTERVAL_TRAIN = config["train"]["inference_interval"]
INFERENCE_INTERVAL_VALID = config["valid"]["inference_interval"]

# Parameters for encoder
C = (
    len(config["data"]["rmic"]["position"]) + 1
)  # Number of physical microphones plus encoded position
L = config["model"]["L"]  # Number of overlapping input segments
F = config["model"]["F"]  # Number of output features of encoder

assert (
    L >= INFERENCE_INTERVAL_TRAIN
), "L must be equal or larger than inference interval!"
assert (
    L >= INFERENCE_INTERVAL_VALID
), "L must be equal or larger than inference interval!"

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

# %% Setup tensorboard and wandb
writer = SummaryWriter(
    os.path.join(config["tensorboard_path"], config["filename"]),
    filename_suffix=".tlog",
)
os.makedirs(CKPT_PATH, exist_ok=True)

wandb.login()
run = wandb.init(
    project=(
        "DeepObservationFilter"
        + ("_pretraining" if not config["train"]["pretrained"] else "")
    ),
    name=config["filename"],
    config=config,
    mode="online" if config["log_wandb"] else "disabled",
)

# %% Set random seed and devices
np.random.seed(config["train"]["seed"])

if torch.cuda.is_available():
    train_device = torch.device("cuda")
    valid_device = torch.device("cuda")
    print("CUDA is available!")

elif torch.backends.mps.is_available():
    train_device = torch.device("mps")
    valid_device = torch.device("mps")
    print("MPS is available!")
else:
    train_device = torch.device("cpu")
    valid_device = torch.device("cpu")
    print("CUDA and MPS are not available!")


# %% Initialize model, optimizer and loss
model_to_train = getattr(models, config["model"]["name"])(config)
model_to_train.to(train_device)

input_fun = lambda inp_shape: {
    "x": torch.FloatTensor(torch.empty(inp_shape)).to(train_device),
    "virt_coords": torch.FloatTensor(torch.empty(1, 3, L)).to(train_device),
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


if config["train"]["optimizer"].lower() == "adam":
    optimizer = torch.optim.Adam(
        model_to_train.parameters(),
        lr=config["train"]["lr_start"],
        weight_decay=config["train"]["weight_decay"],
    )
elif config["train"]["optimizer"].lower() == "adamw":
    optimizer = torch.optim.AdamW(
        model_to_train.parameters(),
        lr=config["train"]["lr_start"],
        weight_decay=config["train"]["weight_decay"],
    )
else:
    sys.exit("Optimizer not supported!")

lr_gamma = np.exp(
    np.log(config["train"]["lr_end"] / config["train"]["lr_start"])
    / config["train"]["lr_epoch"]
)
scheduler = torch.optim.lr_scheduler.ExponentialLR(
    optimizer,
    gamma=lr_gamma,
)

loss_fn = torch.nn.MSELoss()
NMSE = NMSE(return_dB=False, per_sample=True)

# %% Load checkpoint if available
ckpt_file_list = os.listdir(CKPT_PATH)
if len(ckpt_file_list) >= 3:
    # if at least two checkpoint files (model and optimizer) exist => load checkpoints
    latest_model_ckpt, latest_opt_ckpt, latest_sched_ckpt, start_epoch = (
        load_checkpoint(CKPT_PATH, ckpt_file_list, ".ckpt")
    )
    start_epoch += 1
    model_to_train.load_state_dict(latest_model_ckpt)
    optimizer.load_state_dict(latest_opt_ckpt)

    scheduler.load_state_dict(latest_sched_ckpt)

elif os.path.exists(CKPT_PATH_LOAD) and len(os.listdir(CKPT_PATH_LOAD)) >= 3:
    # if pretrained checkpoint path exists => load pretrained checkpoints
    ckpt_file_list = os.listdir(CKPT_PATH_LOAD)
    latest_model_ckpt, latest_opt_ckpt, latest_sched_ckpt, start_epoch = (
        load_checkpoint(CKPT_PATH_LOAD, ckpt_file_list, ".ckpt")
    )
    model_to_train.load_state_dict(latest_model_ckpt)

    print(
        "Pretrained model loaded from "
        + CKPT_PATH_LOAD
        + "! Training starts with epoch #0!"
    )
    start_epoch = 0

else:
    # no checkpoint files are found => create new ones at given directory path
    print("No checkpoints found! Training starts with epoch #0!")
    start_epoch = 0


# %% initialize dataloader
train_ds = DirectionalNoiseDatasetPrerendered(
    config, config["data"]["directory"] + "/train/"
)
valid_ds = DirectionalNoiseDatasetPrerendered(
    config, config["data"]["directory"] + "/valid/"
)

train_dl = data.DataLoader(
    train_ds,
    batch_size=config["train"]["batch_size"],
    num_workers=config["train"]["num_process"],
    shuffle=True,
    drop_last=True,
    collate_fn=custom_collate_fn,
)

valid_dl = data.DataLoader(
    valid_ds,
    batch_size=config["valid"]["batch_size"],
    num_workers=config["valid"]["num_process"],
    shuffle=False,
    drop_last=True,
    collate_fn=custom_collate_fn,
)

N_TRAIN_BATCHES = (
    len(train_dl)
    if config["train"]["n_batches"] == -1
    else config["train"]["n_batches"]
)
N_VALID_BATCHES = (
    len(valid_dl)
    if config["valid"]["n_batches"] == -1
    else config["valid"]["n_batches"]
)

model_to_train.train()

train_loss_per_batch = []
train_loss_per_epoch = []

valid_loss_per_epoch = []
valid_NMSE_per_epoch = []
valid_best_NMSE_per_epoch = []
valid_worst_NMSE_per_epoch = []
valid_median_NMSE_per_epoch = []

valid_epoch_ctr = []
valid_mean_error_fd_per_epoch = []
valid_median_error_fd_per_epoch = []

if VALID_ONLY:
    valid_epoch = start_epoch - 1
else:
    valid_epoch = start_epoch

N_log = config["N_log_epochs"]
ckpt_counter = N_log
log_counter = N_log

OS = OverlapSave(nfft=NFFT, hopsize=HOPSIZE, complex_input=False)

# %% Add one epoch if only validation is performed to trick max epoch setting
if VALID_ONLY:
    EPOCHS += 1

# %% Training loop
for epoch in tqdm.tqdm(
    range(start_epoch, EPOCHS),
    desc="epochs",
    position=0,
):
    train_loss_per_batch = []
    model_to_train.train()

    print(f"Current learning rate: {scheduler.get_last_lr()[0]}")

    for batch, (rm, vm, metadata, vmic_pos) in enumerate(
        tqdm.tqdm(
            train_dl,
            total=N_TRAIN_BATCHES,
            leave=False,
            desc="training batches",
            miniters=int(N_TRAIN_BATCHES / 20),
            disable=True,
        )
    ):
        # Skip training if only validation is performed
        if VALID_ONLY == True:
            break

        if batch >= N_TRAIN_BATCHES:
            break

        # Get signals
        rm = rm.to(train_device)
        vm = vm.to(train_device)
        vmic_pos = vmic_pos.to(train_device)

        # Estimate number of signal blocks
        n_blocks = int(np.floor(rm.shape[-1] / HOPSIZE)) - 1
        n_steps = int(np.floor(n_blocks / INFERENCE_INTERVAL_TRAIN))

        # Convert time to samples for postition data
        vmic_pos[:, 0, :] *= FS
        rm_fd = OS(rm, to_fd=True, reset=True)

        coeffs = torch.zeros(
            (rm.shape[0], C - 1, NFFT // 2 + 1), device=rm.device, dtype=torch.complex64
        )

        # Find start segment index, so that no negative blocks are used
        segment_start = np.ceil(L / INFERENCE_INTERVAL_TRAIN).astype(int) - 1

        for segment in range(segment_start, n_steps):
            start_segment_OA = segment * INFERENCE_INTERVAL_TRAIN
            end_segment = (segment + 1) * INFERENCE_INTERVAL_TRAIN
            start_segment_inf = end_segment - L

            start_sample_inf = (start_segment_inf) * HOPSIZE
            start_sample_vm = (start_segment_OA + 1) * HOPSIZE
            end_sample = (end_segment - 1) * HOPSIZE + NFFT

            if end_sample > vm.shape[2]:
                end_sample = vm.shape[2]

            # Define required time segments to query virtual mic position
            # --> corresponding to last sample in each segment
            t_segments_vmic = (
                torch.arange(start_segment_inf + 1, end_segment + 1).to(train_device)
                * NFFT
                - 1
            )

            # Compensate delay on vm position
            t_segments_vmic += config["model"]["delay"]

            # Find closest position data for each time segment
            virtual_pos = torch.zeros((vmic_pos.shape[0], 3, L), device=train_device)
            for idx, t in enumerate(t_segments_vmic):
                delta_t = vmic_pos[0, 0, :] - t
                # take the highest negative delta_t --> the last position before t
                t_idx = torch.argmax(delta_t[delta_t < 0])

                virtual_pos[:, :, idx] = vmic_pos[:, 1:, t_idx]

            # Perform filter operation
            vm_estimated = OS(
                torch.sum(
                    rm_fd[..., start_segment_OA:end_segment] * coeffs.unsqueeze(-1), 1
                ),
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

            if segment > segment_start:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss_per_batch.append(loss.detach().cpu().numpy())

            # Calculate coefficients for use in next segment
            if segment < n_steps - 1:
                coeffs = model_to_train(
                    rm[..., start_sample_inf:end_sample],
                    virtual_pos,
                )

                # Break if NaN or inf or -inf is found
                if torch.isnan(coeffs).any() or torch.isinf(coeffs).any():
                    print(
                        "NaN, inf or -inf found in coefficients! Stopping training at epoch "
                        + str(epoch)
                    )

    if epoch < config["train"]["lr_epoch"]:
        scheduler.step()
    ckpt_counter -= 1
    log_counter -= 1

    if not (VALID_ONLY):
        if log_counter == 0 or epoch == 0 or epoch == EPOCHS - 1:
            train_loss_per_epoch.append(np.mean(train_loss_per_batch))

            write_loss_summary(
                "training loss (MSE)",
                writer,
                run,
                train_loss_per_epoch[-1],
                epoch,
            )

        # save checkpoint every N_log_ckpt epochs
        if ckpt_counter == 0 or epoch == 0 or epoch == EPOCHS - 1:
            write_checkpoint(
                model_to_train, "DeepObservationfilter_model_", CKPT_PATH, epoch, "ckpt"
            )
            write_checkpoint(
                optimizer, "DeepObservationfilter_optimizer_", CKPT_PATH, epoch, "ckpt"
            )
            write_checkpoint(
                scheduler, "DeepObservationfilter_scheduler_", CKPT_PATH, epoch, "ckpt"
            )

    # Validation
    if log_counter == 0 or VALID_ONLY or epoch == 0 or epoch == EPOCHS - 1:
        valid_epoch_ctr.append(valid_epoch)

        with torch.no_grad():
            valid_loss_per_batch = []

            torch.manual_seed(config["valid"]["seed"] + 2)

            # Initialize lists for samplewise metrics
            metadata_samplewise = []
            valid_NMSE_per_epoch_samplewise = torch.zeros(
                (N_VALID_BATCHES, config["valid"]["batch_size"])
            )
            est_error_psd = np.zeros(
                (
                    N_VALID_BATCHES,
                    config["valid"]["batch_size"],
                    config["valid"]["psd_nfft"] // 2 + 1,
                ),
            )

            # Debug random audio sample per validation
            if config["valid"]["debug_audio"]:
                batch_idx = np.random.randint(0, N_VALID_BATCHES)
                sample_idx = np.random.randint(0, config["valid"]["batch_size"])

            model_to_train.eval()

            for batch, (rm, vm, metadata, vmic_pos) in enumerate(
                tqdm.tqdm(
                    valid_dl,
                    total=N_VALID_BATCHES,
                    leave=False,
                    desc="validation batches",
                    miniters=int(N_VALID_BATCHES / 20),
                    disable=True,
                )
            ):
                if batch >= N_VALID_BATCHES:
                    break
                metadata_samplewise.append(metadata)

                rm = rm.to(train_device)
                vm = vm.to(train_device)
                vmic_pos = vmic_pos.to(train_device)

                # Estimate number of signal blocks
                n_blocks = int(np.floor(rm.shape[-1] / HOPSIZE)) - 1
                n_steps = int(np.floor(n_blocks / INFERENCE_INTERVAL_VALID))

                # Convert time to samples for postition data
                vmic_pos[:, 0, :] *= FS
                rm_fd = OS(rm, to_fd=True, reset=True)

                vm_est_fd_batch = torch.zeros(
                    (rm_fd.shape[0], rm_fd.shape[2], rm_fd.shape[3]),
                    dtype=torch.complex64,
                    requires_grad=False,
                    device=valid_device,
                )

                coeffs = torch.zeros(
                    (rm.shape[0], C - 1, NFFT // 2 + 1),
                    device=rm.device,
                    dtype=torch.complex64,
                )

                # Find start segment index, so that no negative blocks are used
                segment_start = np.ceil(L / INFERENCE_INTERVAL_VALID).astype(int) - 1

                for segment in range(segment_start, n_steps):
                    start_segment_OA = segment * INFERENCE_INTERVAL_VALID
                    end_segment = (segment + 1) * INFERENCE_INTERVAL_VALID
                    start_segment_inf = end_segment - L

                    start_sample_inf = (start_segment_inf) * HOPSIZE
                    start_sample_vm = (start_segment_OA + 1) * HOPSIZE
                    end_sample = (end_segment - 1) * HOPSIZE + NFFT

                    if end_sample > vm.shape[2]:
                        end_sample = vm.shape[2]

                    # Define required time segments to query virtual mic position
                    # --> corresponding to last sample in each segment
                    t_segments_vmic = (
                        torch.arange(start_segment_inf + 1, end_segment + 1).to(
                            train_device
                        )
                        * NFFT
                        - 1
                    )

                    # Compensate delay on vm position if required
                    if config["pos_compensated"]:
                        t_segments_vmic += config["model"]["delay"]

                    # Find closest position data for each time segment
                    virtual_pos = torch.zeros(
                        (vmic_pos.shape[0], 3, L), device=train_device
                    )
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

                    valid_loss_per_batch.append(loss.detach().cpu().numpy())

                    # Calculate coefficients for use in next segment
                    if segment < n_steps - 1:
                        coeffs = model_to_train(
                            rm[..., start_sample_inf:end_sample],
                            virtual_pos,
                        )

                        if torch.isnan(coeffs).any() or torch.isinf(coeffs).any():
                            print(
                                "NaN, inf or -inf found in coefficients! Stopping validation at epoch "
                                + str(epoch)
                            )

                start_sample = L * HOPSIZE
                start_sample_vm = start_sample + HOPSIZE

                vm_estimated = OS(vm_est_fd_batch[..., L:], to_fd=False)
                vm_target = torch.squeeze(vm[..., start_sample_vm:])

                max_vm_length = torch.min(
                    torch.tensor([vm_target.shape[-1], vm_estimated.shape[-1]])
                )

                vm_target = vm_target[..., :max_vm_length]
                vm_estimated = vm_estimated[..., :max_vm_length]

                # Calculate NMSE for each sample in batch
                valid_NMSE_per_epoch_samplewise[batch, :] = NMSE(
                    vm_target, vm_estimated
                )

                # Calculate PSD using torchaudio
                win_psd = torch.hann_window(config["valid"]["psd_nfft"]).to(
                    valid_device
                )
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

                # Debug audio sample
                if config["valid"]["debug_audio"] and batch == batch_idx:
                    write_audio_summary(
                        "validation audio",
                        writer,
                        run,
                        vm_target[sample_idx].cpu(),
                        vm_estimated[sample_idx].cpu(),
                        (vm_target[sample_idx] - vm_estimated[sample_idx]).cpu(),
                        step=epoch,
                        fs=FS,
                    )

            # Save metrics for each sample in batch
            if config["valid"]["export_metrics"]:
                fn = os.path.join(
                    config["valid"]["export_path"],
                    config["filename"] + f"_epoch_{epoch}.npz",
                )

                # Convert metadata dict content to numpy array
                tmp_dict = {}

                for k in metadata_samplewise[0]:
                    tmp_dict[k] = (
                        torch.cat([d[k] for d in metadata_samplewise]).cpu().numpy()
                    )

                np.savez(
                    fn,
                    est_error_psd=est_error_psd.reshape(
                        est_error_psd.shape[0] * est_error_psd.shape[1], -1
                    ),
                    valid_NMSE_per_epoch_samplewise=valid_NMSE_per_epoch_samplewise.flatten()
                    .cpu()
                    .numpy(),
                    metadata=tmp_dict,
                )
            valid_NMSE_per_epoch_samplewise[valid_NMSE_per_epoch_samplewise < 1e-10] = (
                1e-10  # Avoid log(0)
            )
            valid_NMSE_per_epoch_samplewise[
                torch.isnan(valid_NMSE_per_epoch_samplewise)
            ] = 1e10
            valid_NMSE_per_epoch_samplewise = 10 * torch.log10(
                valid_NMSE_per_epoch_samplewise
            )

            # Calculate mean and median NMSE for each sample in batch and show in tensorboard/wandb
            valid_NMSE_per_epoch.append(
                torch.mean(valid_NMSE_per_epoch_samplewise).cpu().numpy()
            )
            valid_best_NMSE_per_epoch.append(
                torch.min(valid_NMSE_per_epoch_samplewise).cpu().numpy()
            )
            valid_worst_NMSE_per_epoch.append(
                torch.max(valid_NMSE_per_epoch_samplewise).cpu().numpy()
            )
            valid_median_NMSE_per_epoch.append(
                torch.median(valid_NMSE_per_epoch_samplewise).cpu().numpy()
            )
            valid_mean_error_fd_per_epoch.append(np.mean(est_error_psd, axis=(0, 1)))
            valid_median_error_fd_per_epoch.append(
                np.median(est_error_psd, axis=(0, 1))
            )
            im_to_plot = plot_spec_tb(
                np.array(valid_mean_error_fd_per_epoch).T,
                epochs=valid_epoch_ctr,
                f_axis=f_axis,
                vmin=-20,
                vmax=10,
                dB=True,
                title="mean validation error",
            )
            write_pyplotfigure_summary(
                "mean validation error spectrogram", writer, run, im_to_plot, epoch
            )
            im_to_plot = plot_spec_singlebatch(
                valid_mean_error_fd_per_epoch[-1],
                f_axis=f_axis,
                vmin=-30,
                vmax=15,
                log_faxis=True,
                dB=True,
                title="mean validation error",
            )
            write_pyplotfigure_summary(
                "mean validation error", writer, run, im_to_plot, epoch
            )
            im_to_plot = plot_spec_tb(
                np.array(valid_median_error_fd_per_epoch).T,
                epochs=valid_epoch_ctr,
                f_axis=f_axis,
                vmin=-20,
                vmax=10,
                dB=True,
                title="median validation error",
            )
            write_pyplotfigure_summary(
                "median validation error spectrogram", writer, run, im_to_plot, epoch
            )
            im_to_plot = plot_spec_singlebatch(
                valid_median_error_fd_per_epoch[-1],
                f_axis=f_axis,
                vmin=-30,
                vmax=15,
                log_faxis=True,
                dB=True,
                title="median validation error",
            )
            write_pyplotfigure_summary(
                "median validation error", writer, run, im_to_plot, epoch
            )

            valid_loss_per_epoch.append(np.mean(valid_loss_per_batch))

            write_loss_summary(
                "validation loss (MSE)", writer, run, valid_loss_per_epoch[-1], epoch
            )

            write_loss_summary(
                "mean validation NMSE in dB",
                writer,
                run,
                valid_NMSE_per_epoch[-1],
                epoch,
            )
            write_loss_summary(
                "median validation NMSE in dB",
                writer,
                run,
                valid_median_NMSE_per_epoch[-1],
                epoch,
            )
            write_loss_summary(
                "minimum validation NMSE in dB",
                writer,
                run,
                valid_worst_NMSE_per_epoch[-1],
                epoch,
            )
            write_loss_summary(
                "maximum validation NMSE in dB",
                writer,
                run,
                valid_best_NMSE_per_epoch[-1],
                epoch,
            )
            write_losshist_summary(
                "validation NMSE in dB",
                writer,
                run,
                valid_NMSE_per_epoch_samplewise.cpu().numpy(),
                epoch,
            )

            del (
                vm_est_fd,
                vm_est_fd_batch,
                vm_estimated,
                vm_target,
                vm_target_stft_for_psd,
                est_error_diff,
                est_error_psd,
                metadata_samplewise,
                psd_target,
                valid_NMSE_per_epoch_samplewise,
                win_psd,
                difference_stft_for_psd,
            )

    if log_counter == 0:
        log_counter = N_log
    if ckpt_counter == 0:
        ckpt_counter = N_log

    valid_epoch += 1

    torch.cuda.empty_cache()

    if VALID_ONLY:
        break

wandb.finish()
print("Training/validation finished!")
