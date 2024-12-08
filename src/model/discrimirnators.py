import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm, weight_norm


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


class PeriodDiscriminator(nn.Module):
    kReluCoef = 0.1

    def __init__(self, period, kernel_size, stride, channels):
        super().__init__()
        self.period = period
        self.convs = nn.ModuleList()
        for i in range(len(channels) - 1):
            self.convs.append(
                weight_norm(
                    nn.Conv2d(
                        channels[i],
                        channels[i + 1],
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(kernel_size, 1), 0),
                    )
                )
            )
        self.convs.append(
            weight_norm(
                nn.Conv2d(
                    channels[-2], channels[-1], (kernel_size, 1), 1, padding=(2, 0)
                )
            )
        )
        self.post = weight_norm(nn.Conv2d(channels[-1], 1, (3, 1), 1, padding=(1, 0)))
        self.lrelu = nn.LeakyReLU(self.kReluCoef)

    def forward(self, x, states):
        tmp = []
        bs, c, t = x.shape
        if t % self.period != 0:
            padding = self.period - (t % self.period)
            x = F.pad(x, (0, padding), "reflect")
            t += padding
        x = x.contiguous().view(bs, c, t // self.period, self.period)

        for conv in self.convs:
            x = self.lrelu(conv(x))
            tmp.append(x)

        x = self.post(x)
        tmp.append(x)
        states.append(tmp)

        return x


class MultiPeriodDiscriminator(nn.Module):
    def __init__(self, periods, kernel_size, stride, channels):
        super().__init__()
        self.discriminators = nn.ModuleList()
        for period in periods:
            self.discriminators.append(
                PeriodDiscriminator(period, kernel_size, stride, channels)
            )

    def forward(self, gt_audio, pr_audio):
        """
        Model forward method.

        Args:
            gt_audio (Tensor): Ground truth audio.
            pr_audio (Tensor): Predicted audio.
        Returns:
            output (tuple): output tuple containing final representations of gt_audio and pr_audio, intermediate representations of them.
        """
        mpd_gt_audio_finals = []
        mpd_pr_audio_finals = []
        mpd_gt_audio_states = []
        mpd_pr_audio_states = []
        for d in self.discriminators:
            gt_audio_final = d(gt_audio, mpd_gt_audio_states)
            pr_audio_final = d(pr_audio, mpd_pr_audio_states)
            mpd_gt_audio_finals.append(gt_audio_final)
            mpd_pr_audio_finals.append(pr_audio_final)

        return (
            mpd_gt_audio_finals,
            mpd_pr_audio_finals,
            mpd_gt_audio_states,
            mpd_pr_audio_states,
        )


class ScaleDiscriminator(nn.Module):
    kReluCoef = 0.1

    def __init__(
        self,
        in_channels,
        out_channels,
        kernels,
        strides,
        paddings,
        groups,
        conv_norm=weight_norm,
    ):
        super().__init__()
        self.convs = nn.ModuleList()
        for in_c, out_c, k, s, p, g in zip(
            in_channels, out_channels, kernels, strides, paddings, groups
        ):
            self.convs.append(conv_norm(nn.Conv1d(in_c, out_c, k, s, p, groups=g)))
        self.lrelu = nn.LeakyReLU(self.kReluCoef)

    def forward(self, x, states):
        bs, c, t = x.shape
        tmp = []
        for i in range(len(self.convs) - 1):
            x = self.lrelu(self.convs[i](x))
            tmp.append(x)
        x = self.convs[-1](x)
        tmp.append(x)
        states.append(tmp)
        return x


class MultiScaleDiscriminator(nn.Module):
    def __init__(self, in_channels, out_channels, kernels, strides, paddings, groups):
        super().__init__()
        self.discriminators = nn.ModuleList(
            [
                ScaleDiscriminator(
                    in_channels,
                    out_channels,
                    kernels,
                    strides,
                    paddings,
                    groups,
                    spectral_norm,
                ),
                ScaleDiscriminator(
                    in_channels,
                    out_channels,
                    kernels,
                    strides,
                    paddings,
                    groups,
                    weight_norm,
                ),
                ScaleDiscriminator(
                    in_channels,
                    out_channels,
                    kernels,
                    strides,
                    paddings,
                    groups,
                    weight_norm,
                ),
            ]
        )
        self.pools = nn.ModuleList([nn.AvgPool1d(4, 2, 2) for _ in range(2)])

    def forward(self, gt_audio, pr_audio):
        """
        Model forward method.

        Args:
            gt_audio (Tensor): Ground truth audio.
            pr_audio (Tensor): Predicted audio.
        Returns:
            output (tuple): output tuple containing final representations of gt_audio and pr_audio, intermediate representations of them.
        """
        msd_gt_audio_states, msd_pr_audio_states = [], []
        gt_audio_final = self.discriminators[0](gt_audio, msd_gt_audio_states)
        pr_audio_final = self.discriminators[0](pr_audio, msd_pr_audio_states)
        msd_gt_audio_finals, msd_pr_audio_finals = [gt_audio_final], [pr_audio_final]

        for i, pool in enumerate(self.pools):
            gt_audio = pool(gt_audio)
            pr_audio = pool(pr_audio)
            gt_audio_final = self.discriminators[i + 1](gt_audio, msd_gt_audio_states)
            pr_audio_final = self.discriminators[i + 1](pr_audio, msd_pr_audio_states)
            msd_gt_audio_finals.append(gt_audio_final)
            msd_pr_audio_finals.append(pr_audio_final)

        return (
            msd_gt_audio_finals,
            msd_pr_audio_finals,
            msd_gt_audio_states,
            msd_pr_audio_states,
        )


class Discriminator(nn.Module):
    def __init__(
        self,
        msd_in_channels,
        msd_out_channels,
        msd_kernels,
        msd_strides,
        msd_paddings,
        msd_groups,
        mpd_periods,
        mpd_kernel_size,
        mpd_stride,
        mpd_channels,
    ):
        super().__init__()
        self.mpd = MultiPeriodDiscriminator(
            mpd_periods, mpd_kernel_size, mpd_stride, mpd_channels
        )
        self.msd = MultiScaleDiscriminator(
            msd_in_channels,
            msd_out_channels,
            msd_kernels,
            msd_strides,
            msd_paddings,
            msd_groups,
        )

    def forward(self, gt_audio, pr_audio, **batch):
        """
        Model forward method.

        Args:
            gt_audio (Tensor): Ground truth audio.
            pr_audio (Tensor): Predicted audio.
        Returns:
            output (dict): output dict containing msd and mpd outputs.
        """
        mpd_output = self.mpd(gt_audio, pr_audio)
        msd_output = self.msd(gt_audio, pr_audio)

        return {"mpd_output": mpd_output, "msd_output": msd_output}

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
