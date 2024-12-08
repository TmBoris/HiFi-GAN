import torch
from torch.nn.functional import l1_loss

from src.metrics.base_metric import BaseMetric


class L1(BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metric = l1_loss

    def __call__(self, gt_spec, pr_spec, **kwargs):
        """
        Metric calculation logic.

        Args:
            gt_spec (Tensor): ground truth MelSpectrogram.
            pr_spec (Tensor): MelSpectrogram of predicted audio.
        Returns:
            metric (float): calculated metric.
        """
        return self.metric(gt_spec, pr_spec)
