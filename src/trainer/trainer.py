import torch

from src.logger.utils import plot_spectrogram
from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer


class Trainer(BaseTrainer):
    """
    Trainer class. Defines the logic of batch logging and processing.
    """

    def process_batch(self, batch, metrics: MetricTracker):
        """
        Run batch through the model, compute metrics, compute loss,
        and do training step (during training stage).

        The function expects that criterion aggregates all losses
        (if there are many) into a single one defined in the 'loss' key.

        Args:
            batch (dict): dict-based batch containing the data from
                the dataloader.
            metrics (MetricTracker): MetricTracker object that computes
                and aggregates the metrics. The metrics depend on the type of
                the partition (train or inference).
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform),
                model outputs, and losses.
        """
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)  # transform batch on device -- faster

        metric_funcs = self.metrics["inference"]
        if self.is_train:
            metric_funcs = self.metrics["train"]
            self.optim_d.zero_grad()

        gen_outputs = self.generator(**batch)
        batch.update(gen_outputs)

        discr_outputs = self.discriminator(
            gt_audio=batch["gt_audio"], pr_audio=batch["pr_audio"].detach()
        )
        batch.update(discr_outputs)

        discr_losses = self.discr_loss(**batch)
        batch.update(discr_losses)

        if self.is_train:
            batch["discr_loss"].backward()  # sum of all losses is always called loss
            self._clip_grad_norm(part="discriminator")
            self.train_metrics.update(
                "discr_grad_norm", self._get_grad_norm(part="discriminator")
            )
            self.optim_d.step()

            self.optim_g.zero_grad()

        discr_outputs = self.discriminator(
            gt_audio=batch["gt_audio"], pr_audio=batch["pr_audio"]
        )
        batch.update(discr_outputs)

        gen_losses = self.gen_loss(**batch)
        batch.update(gen_losses)

        if self.is_train:
            batch["gen_loss"].backward()  # sum of all losses is always called loss
            self._clip_grad_norm(part="generator")
            self.train_metrics.update(
                "gen_grad_norm", self._get_grad_norm(part="generator")
            )
            self.optim_g.step()

        # update metrics for each loss (in case of multiple losses)
        for loss_name in self.config.writer.loss_names:
            metrics.update(loss_name, batch[loss_name].item())

        for met in metric_funcs:
            metrics.update(met.name, met(**batch))
        return batch

    def _log_batch(self, batch_idx, batch, mode="train"):
        """
        Log data from batch. Calls self.writer.add_* to log data
        to the experiment tracker.

        Args:
            batch_idx (int): index of the current batch.
            batch (dict): dict-based batch after going through
                the 'process_batch' function.
            mode (str): train or inference. Defines which logging
                rules to apply.
        """
        # method to log data from you batch
        # such as audio, text or images, for example

        # logging scheme might be different for different partitions
        if mode == "train":  # the method is called only every self.log_step steps
            self.log_predictions(**batch)
            self.log_spectrogram(**batch)
        else:
            self.log_predictions(**batch)
            self.log_spectrogram(**batch)

    def log_spectrogram(self, gt_spec, pr_spec, **batch):
        spectrogram_for_plot = gt_spec[0].detach().cpu()
        image = plot_spectrogram(spectrogram_for_plot)
        self.writer.add_image("gt_spec", image)

        spectrogram_for_plot = pr_spec[0].detach().cpu()
        image = plot_spectrogram(spectrogram_for_plot)
        self.writer.add_image("pr_spec", image)

    def log_predictions(self, gt_audio, pr_audio, **batch):
        # print('gt_audio.shape', gt_audio.shape)
        # print('pr_audio.shape', pr_audio.shape)

        self.writer.add_audio("gt_audio", gt_audio[0], 22050)
        self.writer.add_audio("pr_audio", pr_audio[0], 22050)
