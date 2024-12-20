import torch
import torch.nn.functional as F
from torch import nn


class GeneratorLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, mpd_output, msd_output, gt_spec, pr_spec, **batch):
        """
        Generator Loss calculation logic.

        Args:
            mpd_output (tuple): MultyPeriodDiscriminator output.
            msd_output (tuple): MultyScaleDiscriminator output.
            gt_spec (Tensor): ground truth MelSpectrogram.
            pr_spec (Tensor): MelSpectrogram of predicted audio.
        Returns:
            losses (dict): dict containing calculated loss functions.
        """
        _, mpd_pr_audio_finals, mpd_gt_audio_states, mpd_pr_audio_states = mpd_output
        _, msd_pr_audio_finals, msd_gt_audio_states, msd_pr_audio_states = msd_output

        mpd_gen_loss = self.generator_loss(mpd_pr_audio_finals)
        msd_gen_loss = self.generator_loss(msd_pr_audio_finals)
        adv_gen_loss = mpd_gen_loss + msd_gen_loss

        mpd_feature_loss = self.feature_loss(mpd_gt_audio_states, mpd_pr_audio_states)
        msd_feature_loss = self.feature_loss(msd_gt_audio_states, msd_pr_audio_states)
        feature_loss = mpd_feature_loss + msd_feature_loss

        loss_mel = F.l1_loss(gt_spec, pr_spec)

        loss = adv_gen_loss + 2 * feature_loss + 45 * loss_mel

        return {
            "gen_loss": loss,
            "adv_gen_loss": adv_gen_loss,
            "feature_loss": feature_loss,
            "mel_loss": loss_mel,
        }

    def feature_loss(self, gt_audio_states_list, pr_audio_states_list):
        loss = 0
        for gt_audio_states, pr_audio_states in zip(
            gt_audio_states_list, pr_audio_states_list
        ):
            for gt_audio_state, pr_audio_state in zip(gt_audio_states, pr_audio_states):
                loss += F.l1_loss(gt_audio_state, pr_audio_state)
        return loss

    def generator_loss(self, pr_audio_finals):
        loss = 0
        for pr_audio_final in pr_audio_finals:
            loss += F.mse_loss(pr_audio_final, torch.ones_like(pr_audio_final))
        return loss
