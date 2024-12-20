import torch
import torch.nn as nn
import torch.nn.functional as F
from hydra.utils import instantiate
from torch.nn.utils import weight_norm


def init_weights(m, mean=0.0, std=1e-2):
    m.weight.data.normal_(mean, std)


class MRF(nn.Module):
    kReluCoef = 0.1

    def __init__(self, channels, kernel_size, dilations):
        super().__init__()
        self.conv1 = nn.ModuleList()
        self.conv2 = nn.ModuleList()
        for dilation in dilations:
            self.conv1.append(
                weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        dilation=dilation,
                        padding="same",
                    )
                ).apply(init_weights)
            )
            self.conv2.append(
                nn.Conv1d(
                    channels, channels, kernel_size, dilation=1, padding="same"
                ).apply(init_weights)
            )

        self.lrelu = nn.LeakyReLU(self.kReluCoef)

    def forward(self, x):
        for i in range(len(self.conv1)):
            resid = x
            x = self.conv1[i](self.lrelu(x))
            resid = resid + x
            x = self.conv2[i](self.lrelu(resid))
            x = x + resid
        return x


class Generator(nn.Module):
    kReluCoef = 0.1

    def __init__(
        self,
        mrf_kernel_sizes,
        mrf_dilation_sizes,
        get_spectrogram,
        upsample_initial_channel,
        upsample_kernel_sizes,
        prepost_conv_kernel_size,
        n_mels,
    ):
        super().__init__()
        self.num_kernels = len(mrf_kernel_sizes)
        self.pre_conv = weight_norm(
            nn.Conv1d(
                n_mels,
                upsample_initial_channel,
                prepost_conv_kernel_size,
                padding="same",
            )
        )

        self.ups = nn.ModuleList()
        self.mrfs = nn.ModuleList()

        in_c = upsample_initial_channel * 2
        for kernel_size in upsample_kernel_sizes:
            in_c //= 2
            out_c = in_c // 2
            stride = kernel_size // 2
            self.ups.append(
                weight_norm(
                    nn.ConvTranspose1d(
                        in_c,
                        out_c,
                        kernel_size,
                        stride,
                        padding=(kernel_size - stride) // 2,
                    )
                ).apply(init_weights)
            )

            for kernel_size, dilations in zip(mrf_kernel_sizes, mrf_dilation_sizes):
                self.mrfs.append(MRF(out_c, kernel_size, dilations))

        self.get_spectrogram = get_spectrogram
        self.lrelu1 = nn.LeakyReLU(self.kReluCoef)

        self.lrelu2 = nn.LeakyReLU()
        self.post_conv = weight_norm(
            nn.Conv1d(out_c, 1, prepost_conv_kernel_size, padding="same")
        ).apply(init_weights)

        self.tanh = nn.Tanh()

    def forward(self, gt_spec, **batch):
        """
        Model forward method.

        Args:
            gt_spec (Tensor): Ground truth MelSpectrogram.
        Returns:
            output (tuple): output dict containing predicted audio and its MelSpectrogram.
        """
        x = self.pre_conv(gt_spec)

        for i in range(len(self.ups)):
            x = self.ups[i](self.lrelu1(x))
            # mrf
            x_summator = self.mrfs[i * self.num_kernels](x)
            for j in range(self.num_kernels):
                x_summator += self.mrfs[i * self.num_kernels + j](x)

        x = self.tanh(self.post_conv(self.lrelu2(x_summator)))

        return {
            "pr_audio": x,
            "pr_spec": torch.log(self.get_spectrogram(x).squeeze(1) + 1e-5),
        }

    def __str__(self):
        """
        Model prints with the number of parameters.
        """
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )

        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"

        return result_info
