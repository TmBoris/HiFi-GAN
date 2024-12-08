import torch
import torch.nn.functional as F
from torch import nn


class DiscriminatorLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, mpd_output, msd_output, **batch):
        """
        Discriminator Loss calculation logic.

        Args:
            mpd_output (tuple): MultyPeriodDiscriminator output.
            msd_output (tuple): MultyScaleDiscriminator output.
        Returns:
            losses (dict): dict containing calculated loss functions.
        """
        assert len(mpd_output) == 4, "wrong shape of mpd_output"
        assert len(msd_output) == 4, "wrong shape of msd_output"

        mpd_gt_audio_finals, mpd_pr_audio_finals, _, _ = mpd_output
        msd_gt_audio_finals, msd_pr_audio_finals, _, _ = msd_output
        loss_disc_mpd = self.discriminator_loss(
            mpd_gt_audio_finals, mpd_pr_audio_finals
        )
        loss_disc_msd = self.discriminator_loss(
            msd_gt_audio_finals, msd_pr_audio_finals
        )

        return {
            "discr_loss": loss_disc_msd + loss_disc_mpd,
            "loss_disc_mpd": loss_disc_mpd,
            "loss_disc_msd": loss_disc_msd,
        }

    def discriminator_loss(self, gt_audio_finals, pr_audio_finals):
        loss = 0

        for gt_audio_final, pr_audio_final in zip(gt_audio_finals, pr_audio_finals):
            gt_audio_loss = F.mse_loss(gt_audio_final, torch.ones_like(gt_audio_final))
            pr_audio_loss = F.mse_loss(pr_audio_final, torch.zeros_like(pr_audio_final))
            loss += gt_audio_loss + pr_audio_loss

        return loss
