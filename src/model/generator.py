import torch
import torch.nn as nn
import torch.nn.functional as F
from hydra.utils import instantiate
from torch.nn.utils import weight_norm


def init_weights(m, mean=0.0, std=1e-2):
    m.weight.data.normal_(mean, std)


class ResBlock(nn.Module):
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
            x = self.conv2[i](self.lrelu(x))
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
            n_mels
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
        self.resblocks = nn.ModuleList()

        in_c = upsample_initial_channel * 2
        for kernel_size in upsample_kernel_sizes:
            in_c //= 2
            out_c = in_c // 2
            stride = kernel_size // 2
            self.ups.append(weight_norm(
                nn.ConvTranspose1d(
                    in_c,
                    out_c,
                    kernel_size,
                    stride,
                    padding=(kernel_size - stride) // 2,
                )
            ).apply(init_weights))

            for kernel_size, dilations in zip(mrf_kernel_sizes, mrf_dilation_sizes):
                self.resblocks.append(ResBlock(out_c, kernel_size, dilations))

        self.get_spectrogram = get_spectrogram
        self.lrelu1 = nn.LeakyReLU(self.kReluCoef)

        self.post = nn.Sequential(
            nn.LeakyReLU(),
            weight_norm(
                nn.Conv1d(out_c, 1, prepost_conv_kernel_size, padding="same")
            ).apply(init_weights),
            nn.Tanh()
        )

    def forward(self, gt_spec, **batch):
        x = self.pre_conv(gt_spec)

        for i in range(len(self.ups)):
            x = self.ups[i](self.lrelu1(x))
            # mrf
            x_summator = self.resblocks[i * self.num_kernels](x)
            for j in range(self.num_kernels):
                x_summator += self.resblocks[i * self.num_kernels + j](x)
            x_means = x_summator / self.num_kernels

        x = self.post(x_means)

        return {
            "pr_audio": x,
            "pr_spec": torch.log(self.get_spectrogram(x).squeeze(1) + 1e-5),
        }
