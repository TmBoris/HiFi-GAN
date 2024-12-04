import torch
from torch import nn


class DiscriminatorLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, mpd_output, msd_output, **batch):
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
            gt_audio_loss = torch.mean((1 - gt_audio_final) ** 2)
            pr_audio_loss = torch.mean(pr_audio_final ** 2)
            loss += gt_audio_loss + pr_audio_loss

        return loss
