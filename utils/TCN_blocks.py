import torch


class TCNConvBlock(torch.nn.Module):
    """Single dilated convolutional block used within a TCN stack.

    Applies a 1x1 input convolution, depthwise separable dilated 2-D
    convolution, and produces both a residual and a skip-connection output.
    """

    def __init__(self, C_b, C_TCN, dilation, kernelsize, padding):
        """Initialize TCNConvBlock.

        Parameters
        ----------
        C_b : int
            Number of bottleneck channels (input and output).
        C_TCN : int
            Number of hidden channels inside the block.
        dilation : int
            Dilation factor for the depthwise convolution.
        kernelsize : int or list of int
            Kernel size for the depthwise convolution.
        padding : int
            Padding applied along the time axis.
        """
        super().__init__()

        if isinstance(kernelsize, int):
            _kernelsize = (kernelsize, kernelsize)
        elif isinstance(kernelsize, (list, tuple)):
            _kernelsize = tuple(kernelsize)
        else:
            raise ValueError("Invalid kernel size")

        conv_input = torch.nn.Conv2d(C_b, C_TCN, (1, 1))
        self.conv_output = torch.nn.Conv2d(C_TCN, C_b, (1, 1))
        self.conv_skip = torch.nn.Conv2d(C_TCN, C_b, (1, 1))

        act1 = torch.nn.PReLU()
        act2 = torch.nn.PReLU()

        norm1 = torch.nn.GroupNorm(1, C_TCN, eps=1e-8)
        norm2 = torch.nn.GroupNorm(1, C_TCN, eps=1e-8)

        dconv2d = torch.nn.Conv2d(
            C_TCN,
            C_TCN,
            _kernelsize,
            dilation=(1, dilation),
            groups=C_TCN,
            padding=(1, padding),
        )

        self.net_main = torch.nn.Sequential(
            conv_input,
            act1,
            norm1,
            dconv2d,
            act2,
            norm2,
        )

    def forward(self, x):
        """Forward pass through the TCN convolution block.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, C_b, F_b, L).

        Returns
        -------
        tuple of torch.Tensor
            Residual output and skip-connection output, each of shape
            (batch, C_b, F_b, L).
        """
        residual = x

        x = self.net_main(x)

        residual = residual + self.conv_output(x)
        x = self.conv_skip(x)

        return residual, x


class TNC_noBN(torch.nn.Module):
    """Temporal Convolutional Network for observation filter estimation.

    Applies feature and channel bottleneck layers followed by stacked dilated
    TCN convolution blocks with skip connections, and a linear output layer.
    """

    def __init__(self, C, C_b, F, F_b, L, K, C_TCN, D, S, TCN_kernelsize):
        """Initialize TCN.

        Parameters
        ----------
        C : int
            Number of input channels (microphones + position).
        C_b : int
            Number of bottleneck channels.
        F : int
            Number of encoder features.
        F_b : int
            Number of bottleneck features.
        L : int
            Number of overlapping input frames.
        K : int
            Output filter length in time domain.
        C_TCN : int
            Number of hidden channels in TCN convolution blocks.
        D : int
            Number of dilated layers per TCN stack.
        S : int
            Number of TCN stacks.
        TCN_kernelsize : int or list of int
            Kernel size of TCN convolution blocks.
        """
        super().__init__()

        self.norm = torch.nn.GroupNorm(1, F, eps=1e-8)

        self.bottleneck1 = torch.nn.Conv2d(F, F_b, kernel_size=(1, 1))
        self.bottleneck2 = torch.nn.Conv2d(C, C_b, kernel_size=(1, 1))

        self.net_main = torch.nn.ModuleList([])

        for stack in range(S):
            for layer in range(D):
                self.net_main.append(
                    TCNConvBlock(C_b, C_TCN, 2**layer, TCN_kernelsize, 2**layer)
                )

        self.act_out = torch.nn.PReLU()
        self.conv_out1 = torch.nn.Conv2d(C_b, C - 1, kernel_size=(1, 1))

        self.out = torch.nn.Linear(L * F_b, K)

    def forward(self, x):
        """Forward pass through the TCN.

        Parameters
        ----------
        x : torch.Tensor
            Encoded input of shape (batch, C, F, L).

        Returns
        -------
        torch.Tensor
            Estimated filter coefficients of shape (batch, C-1, K).
        """
        batchsize, C, F, L = x.shape

        x = self.norm(x.view(batchsize * C, F, L)).view(batchsize, C, F, L)
        x = self.bottleneck1(x.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
        x = self.bottleneck2(x)

        skip = torch.zeros_like(x)

        for layer in self.net_main:
            x, skip_buffer = layer(x)
            skip = skip + skip_buffer

        skip = self.act_out(skip)
        skip = self.conv_out1(skip)

        skip = self.out(skip.view(batchsize * (C - 1), -1)).view(batchsize, C - 1, -1)

        return skip


class TCN(torch.nn.Module):
    """Modified TCN with an additional bottleneck along the time axis.

    Extends the base TCN by adding a time-domain bottleneck layer that reduces
    the temporal dimension before the dilated convolution stacks.
    """

    def __init__(self, C, C_b, F, F_b, L, L_b, K, C_TCN, D, S, TCN_kernelsize):
        """Initialize TCN_mod.

        Parameters
        ----------
        C : int
            Number of input channels (microphones + position).
        C_b : int
            Number of bottleneck channels.
        F : int
            Number of encoder features.
        F_b : int
            Number of bottleneck features.
        L : int
            Number of overlapping input frames.
        L_b : int
            Length of the bottleneck time context.
        K : int
            Output filter length in time domain.
        C_TCN : int
            Number of hidden channels in TCN convolution blocks.
        D : int
            Number of dilated layers per TCN stack.
        S : int
            Number of TCN stacks.
        TCN_kernelsize : int or list of int
            Kernel size of TCN convolution blocks.
        """
        super().__init__()

        self.norm = torch.nn.GroupNorm(1, F, eps=1e-8)

        self.bottleneck1 = torch.nn.Conv2d(F, F_b, kernel_size=(1, 1))
        self.bottleneck2 = torch.nn.Conv2d(C, C_b, kernel_size=(1, 1))
        self.bottleneck3 = torch.nn.Conv2d(L, L_b, kernel_size=(1, 1))

        self.net_main = torch.nn.ModuleList([])

        for stack in range(S):
            for layer in range(D):
                self.net_main.append(
                    TCNConvBlock(C_b, C_TCN, 2**layer, TCN_kernelsize, 2**layer)
                )

        self.act_out = torch.nn.PReLU()
        self.conv_out1 = torch.nn.Conv2d(C_b, C - 1, kernel_size=(1, 1))
        self.out = torch.nn.Linear(L_b * F_b, K)

    def forward(self, x):
        """Forward pass through the modified TCN.

        Parameters
        ----------
        x : torch.Tensor
            Encoded input of shape (batch, C, F, L).

        Returns
        -------
        torch.Tensor
            Estimated filter coefficients of shape (batch, C-1, K).
        """
        batchsize, C, F, L = x.shape

        x = self.norm(x.view(batchsize * C, F, L)).view(batchsize, C, F, L)
        x = self.bottleneck1(x.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
        x = self.bottleneck2(x)
        x = self.bottleneck3(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

        skip = torch.zeros_like(x)

        for layer in self.net_main:
            x, skip_buffer = layer(x)
            skip = skip + skip_buffer

        skip = self.act_out(skip)
        skip = self.conv_out1(skip)

        skip = self.out(skip.view(batchsize * (C - 1), -1)).view(batchsize, C - 1, -1)

        return skip
