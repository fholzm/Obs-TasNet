import torch
from utils.transforms import OverlapSave
from utils.TCN_blocks import TNC_noBN, TCN


class ObsTasNet_noBN(torch.nn.Module):
    """Observation filter estimation based on IC Conv-TasNet without temporal bottleneck.

    Uses a 1-D convolutional encoder, position encoding, and a TCN backbone
    to estimate observation filter coefficients in the frequency domain.
    """

    def __init__(self, config):
        """Initialize ObsTasNet_noBN.

        Parameters
        ----------
        config : dict
            Configuration dictionary containing model, data, and signal
            processing parameters.
        """
        super().__init__()

        # Define all necessary constants
        self.fs = config["samplerate"]
        self.K = config.get("n_coeffs", config["nfft"] - config["hopsize"] + 1)
        self.nfft = config["nfft"]
        self.hopsize = config["hopsize"]

        # Parameters for encoder
        self.C = (
            len(config["data"]["rmic"]["position"]) + 1
        )  # Number of physical microphones plus encoded position
        self.W = config["nfft"]  # Length of window on input signal
        self.L = config["model"]["L"]  # Number of overlapping input segments
        self.F = config["model"]["F"]  # Number of encoder features

        # Parameters for bottleneck
        self.F_b = config["model"]["F_b"]  # Feature dimension after bottleneck layer
        self.C_b = config["model"]["C_b"]  # Channel dimension after bottleneck layer
        self.C_TCN = config["model"]["C_TCN"]  # Channels in the TCN convolution blocks
        self.D = config["model"]["D"]  # Number of TCN convolution blocks per stack
        self.TCN_kernelsize = config["model"]["TCN_kernelsize"]  # Kernel size of TCN
        self.S = config["model"]["S"]  # Number of TCN stacks

        # OS
        self.OS = OverlapSave(
            nfft=self.nfft,
            hopsize=self.hopsize,
            complex_input=False,
        )

        # Neural network
        self.encoder = torch.nn.Conv1d(
            in_channels=1,
            out_channels=self.F,
            kernel_size=self.W,
            bias=False,
            stride=self.hopsize,
        )

        self.encoder_position = torch.nn.Conv1d(
            in_channels=3,
            out_channels=self.F,
            kernel_size=1,
            bias=False,
            stride=1,
        )

        self.TCN = TNC_noBN(
            self.C,
            self.C_b,
            self.F,
            self.F_b,
            self.L,
            self.K,
            self.C_TCN,
            self.D,
            self.S,
            self.TCN_kernelsize,
        )

    def forward(
        self,
        x,
        virt_coords,
    ):
        """Estimate observation filter coefficients.

        Parameters
        ----------
        x : torch.Tensor
            Microphone input signals of shape (batch, channels, samples).
        virt_coords : torch.Tensor
            Virtual microphone coordinates of shape (batch, 3, L).

        Returns
        -------
        torch.Tensor
            Estimated filter coefficients in frequency domain.
        """
        if x.ndim == 2:
            x = x.unsqueeze(0)

        assert x.shape[0] == virt_coords.shape[0], "Batch size mismatch."
        assert x.shape[-1] == self.W + self.hopsize * (
            self.L - 1
        ), "Input length mismatch."
        # Calculate batch size, number of blocks in batch, and number of optimization steps
        batch_size = x.shape[0]

        x = self.encoder(x.view(batch_size * (self.C - 1), 1, -1)).view(
            batch_size, self.C - 1, self.F, self.L
        )

        # Encode position
        virt_pos_tmp = self.encoder_position(virt_coords).unsqueeze(1)

        # Concatenate encoded inputs and position
        x = torch.cat((x, virt_pos_tmp), dim=1)

        x = self.TCN(x)

        # Apply TCN
        return torch.fft.rfft(x, n=self.nfft)


class ObsTasNet(torch.nn.Module):
    """Observation filter estimation based on modified IC Conv-TasNet with temporal bottleneck layer.

    Uses a 1-D convolutional encoder, position encoding, and a modified TCN
    backbone with bottleneck time context to estimate observation filter
    coefficients in the frequency domain.
    """

    def __init__(self, config):
        """Initialize ObsTasNet.

        Parameters
        ----------
        config : dict
            Configuration dictionary containing model, data, and signal
            processing parameters.
        """
        super().__init__()

        # Define all necessary constants
        self.fs = config["samplerate"]
        self.K = config.get("n_coeffs", config["nfft"] - config["hopsize"] + 1)
        self.nfft = config["nfft"]
        self.hopsize = config["hopsize"]

        # Parameters for encoder
        self.C = (
            len(config["data"]["rmic"]["position"]) + 1
        )  # Number of physical microphones plus encoded position
        self.W = config["nfft"]  # Length of window on input signal
        self.L = config["model"]["L"]  # Number of overlapping input segments
        self.L_b = config["model"]["L_b"]  # Length of bottleneck time context
        self.F = config["model"]["F"]  # Number of encoder features

        # Parameters for bottleneck
        self.F_b = config["model"]["F_b"]  # Feature dimension after bottleneck layer
        self.C_b = config["model"]["C_b"]  # Channel dimension after bottleneck layer
        self.C_TCN = config["model"]["C_TCN"]  # Channels in the TCN convolution blocks
        self.D = config["model"]["D"]  # Number of TCN convolution blocks per stack
        self.TCN_kernelsize = config["model"]["TCN_kernelsize"]  # Kernel size of TCN
        self.S = config["model"]["S"]  # Number of TCN stacks

        # OS
        self.OS = OverlapSave(
            nfft=self.nfft,
            hopsize=self.hopsize,
            complex_input=False,
        )

        # Neural network
        self.encoder = torch.nn.Conv1d(
            in_channels=1,
            out_channels=self.F,
            kernel_size=self.W,
            bias=False,
            stride=self.hopsize,
        )

        self.encoder_position = torch.nn.Conv1d(
            in_channels=3,
            out_channels=self.F,
            kernel_size=1,
            bias=False,
            stride=1,
        )

        self.TCN = TCN(
            self.C,
            self.C_b,
            self.F,
            self.F_b,
            self.L,
            self.L_b,
            self.K,
            self.C_TCN,
            self.D,
            self.S,
            self.TCN_kernelsize,
        )

    def forward(
        self,
        x,
        virt_coords,
    ):
        """Estimate observation filter coefficients.

        Parameters
        ----------
        x : torch.Tensor
            Microphone input signals of shape (batch, channels, samples).
        virt_coords : torch.Tensor
            Virtual microphone coordinates of shape (batch, 3, L).

        Returns
        -------
        torch.Tensor
            Estimated filter coefficients in frequency domain, zero-padded
            to nfft length before the FFT.
        """
        if x.ndim == 2:
            x = x.unsqueeze(0)

        assert x.shape[0] == virt_coords.shape[0], "Batch size mismatch."
        assert x.shape[-1] == self.W + self.hopsize * (
            self.L - 1
        ), "Input length mismatch."

        # Calculate batch size, number of blocks in batch, and number of optimization steps
        batch_size = x.shape[0]

        x = self.encoder(x.view(batch_size * (self.C - 1), 1, -1)).view(
            batch_size, self.C - 1, self.F, self.L
        )

        # Encode position
        virt_pos_tmp = self.encoder_position(virt_coords).unsqueeze(1)

        # Concatenate encoded inputs and position
        x = torch.cat((x, virt_pos_tmp), dim=1)

        x = self.TCN(x)
        x_shape = list(x.shape)

        x = torch.concat(
            [
                x,
                torch.zeros(
                    x_shape[0], x_shape[1], self.nfft - x_shape[2], device=x.device
                ),
            ],
            dim=2,
        )

        # Apply TCN
        return torch.fft.rfft(x, n=self.nfft)
