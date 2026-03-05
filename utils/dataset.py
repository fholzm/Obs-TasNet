import glob
import torch, torchaudio
import pandas as pd
import warnings

# Suppress specific torchaudio warning
warnings.filterwarnings(
    "ignore",
    message="In 2.9, this function's implementation will be changed to use torchaudio.load_with_torchcodec",
)


class DirectionalNoiseDatasetPrerendered(torch.utils.data.Dataset):
    """Class for the Directional Noise Dataset of pre-rendered scenes."""

    def __init__(self, config: dict, directory: str):
        """Class for the Directional Noise Dataset of pre-rendered scenes.

        Parameters
        ----------
        config : dict
            Config dictionary for dataset parameters
        directory : str
            Directory containing the pre-rendered scenes

        Attributes
        ----------
        directory : str
            Directory containing the pre-rendered scenes
        n_remotemics : int
            Number of remote microphones in the dataset
        n_virtualmics : int
            Number of virtual microphones in the dataset (currently set to 1)
        delay : int
            Delay in samples for the remote microphone signals
        nfft : int
            FFT size for processing the signals
        hopsize : int
            Hopsize for processing the signals
        samplelength : int
            Length of the audio samples in the dataset (calculated from the first scene)
        metadata : pd.DataFrame
            Metadata for the scenes in the dataset, loaded from a pickle file
        """
        self.directory = directory
        self.n_remotemics = len(config["data"]["rmic"]["position"])
        self.n_virtualmics = 1
        self.delay = config["model"]["delay"]
        self.nfft = config["nfft"]
        self.hopsize = config["hopsize"]

        sample_data, _ = torchaudio.load(self.directory + f"scene_0.wav")
        self.samplelength = sample_data.shape[-1] - self.delay
        self.metadata = pd.read_pickle(self.directory + "metadata.pkl")

    def __len__(self):
        """Return the number of scenes in the dataset.

        Returns
        -------
        int
            Number of scene files in the dataset directory
        """
        return len(glob.glob1(self.directory, "scene*.wav"))

    def __getitem__(self, index):
        """Load and return a single sample from the dataset.

        Parameters
        ----------
        index : int
            Index of the scene to load

        Returns
        -------
        tuple
            Tuple containing (rm_signal, vm_signal, metadata, vmic_positions) where:
            - rm_signal: Tensor of remote microphone signals
            - vm_signal: Tensor of virtual microphone signals
            - metadata: Metadata dictionary for the scene
            - vmic_positions: Tensor of virtual microphone positions
        """
        signal, _ = torchaudio.load(self.directory + f"scene_{index}.wav")

        rm_signal = signal[
            : self.n_remotemics, self.delay : self.samplelength + self.delay
        ]
        vm_signal = signal[
            self.n_remotemics : self.n_remotemics + self.n_virtualmics,
            0 : self.samplelength,
        ]

        csv_fn = self.directory + f"vmic_pos_{index}.csv"
        position_data = pd.read_csv(csv_fn, header=None)
        vmic_positions = torch.tensor(position_data.values, dtype=torch.float32).T

        return rm_signal, vm_signal, self.metadata[index], vmic_positions


def custom_collate_fn(batch):
    """Custom collate function for the Directional Noise Dataset of pre-rendered scenes to stack audio signals and metadata into tensors and lists, respectively.

    Parameters
    ----------
    batch : list
         List of samples from the dataset, where each sample is a tuple of (rm_signal, vm_signal, metadata, vmic_positions)

    Returns
    -------
    tuple
        Tuple containing stacked rm_signals, vm_signals, list of metadata, and stacked vmic_positions
    """
    rm_signals, vm_signals, metadata, vmic_positions = zip(*batch)

    # Stack audio signals into tensors
    rm_signals = torch.stack(rm_signals)
    vm_signals = torch.stack(vm_signals)
    vmic_positions = torch.stack(vmic_positions)

    return rm_signals, vm_signals, list(metadata), vmic_positions
