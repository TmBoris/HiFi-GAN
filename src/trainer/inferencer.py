import nltk
import torch
import torchaudio
from hydra.utils import instantiate
from torch import nn
from tqdm.auto import tqdm

from src.logger.utils import plot_spectrogram
from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer


class Inferencer(BaseTrainer):
    """
    Inferencer (Like Trainer but for Inference) class

    The class is used to process data without
    the need of optimizers, writers, etc.
    Required to evaluate the model on the dataset, save predictions, etc.
    """

    def __init__(
        self,
        generator,
        config,
        device,
        samples,
        save_path,
        text_to_mel_model="fastspeech2",
        writer=None,
        resynthesize=False,
        skip_model_load=False,
    ):
        """
        Initialize the Inferencer.

        Args:
            model (nn.Module): PyTorch model.
            config (DictConfig): run config containing inferencer config.
            device (str): device for tensors and model.
            dataloaders (dict[DataLoader]): dataloaders for different
                sets of data.
            save_path (str): path to save model predictions and other
                information.
            metrics (dict): dict with the definition of metrics for
                inference (metrics[inference]). Each metric is an instance
                of src.metrics.BaseMetric.
            batch_transforms (dict[nn.Module] | None): transforms that
                should be applied on the whole batch. Depend on the
                tensor name.
            skip_model_load (bool): if False, require the user to set
                pre-trained checkpoint path. Set this argument to True if
                the model desirable weights are defined outside of the
                Inferencer Class.
        """
        assert (
            skip_model_load or config.inferencer.get("from_pretrained") is not None
        ), "Provide checkpoint or set skip_model_load=True"

        self.config = config
        self.cfg_trainer = self.config.inferencer

        self.device = device
        self.resynthesize = resynthesize
        self.generator = nn.DataParallel(generator)
        self.samples = samples
        self.writer = writer
        # path definition
        self.save_path = save_path
        self.text_to_mel_model_type = text_to_mel_model

        if not self.resynthesize:
            if self.text_to_mel_model_type == "fastspeech2":
                self.tokenizer = torch.load("text_to_mel_tokenizer.pt")
                nltk.download("averaged_perceptron_tagger_eng")
                self.text_to_mel_model = torch.load(
                    "text_to_mel_model.pt", map_location=self.device
                )
            elif self.text_to_mel_model_type == "tacotron2":
                self.tacotron2 = torch.hub.load(
                    "NVIDIA/DeepLearningExamples:torchhub",
                    "nvidia_tacotron2",
                    model_math="fp16",
                ).to(self.device)
                self.tacotron2.eval()
                self.tacotron2._modules["decoder"].max_decoder_steps = 5000
                self.taco_utils = torch.hub.load(
                    "NVIDIA/DeepLearningExamples:torchhub", "nvidia_tts_utils"
                )
            else:
                raise ValueError(
                    "Only fastspeech2 or tacotron2 for text_to_mel_model available"
                )
        else:
            self.get_spec = instantiate(self.config.get_spectrogram).to(self.device)

        if not skip_model_load:
            # init model
            self._from_pretrained(config.inferencer.get("from_pretrained"))

    def run_inference(self):
        """
        Run inference on each partition.

        Returns:
            part_logs (dict): part_logs[part_name] contains logs
                for the part_name partition.
        """
        self._inference_part()

    def _inference_part(self):
        """
        Run inference on a given partition and save predictions

        Args:
            part (str): name of the partition.
            dataloader (DataLoader): dataloader for the given partition.
        Returns:
            logs (dict): metrics, calculated on the partition.
        """

        self.is_train = False
        self.generator.eval()

        # create Save dir
        if self.save_path is not None:
            if self.resynthesize:
                (self.save_path / "gt_audio").mkdir(exist_ok=True, parents=True)
            (self.save_path / "pr_audio").mkdir(exist_ok=True, parents=True)

        with torch.no_grad():
            for batch_idx, batch in tqdm(
                enumerate(self.samples),
                total=len(self.samples),
            ):
                batch = self.process_batch(
                    batch_idx=batch_idx,
                    batch=batch,
                )

    def process_batch(self, batch_idx, batch):
        """
        Run batch through the model, compute metrics, and
        save predictions to disk.

        Save directory is defined by save_path in the inference
        config and current partition.

        Args:
            batch_idx (int): the index of the current batch.
            batch (dict): dict-based batch containing the data from
                the dataloader.
            metrics (MetricTracker): MetricTracker object that computes
                and aggregates the metrics. The metrics depend on the type
                of the partition (train or inference).
            part (str): name of the partition. Used to define proper saving
                directory.
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform)
                and model outputs.
        """
        batch = self.move_batch_to_device(batch)

        if not self.resynthesize:
            if self.text_to_mel_model_type == "fastspeech2":
                inputs = self.tokenizer(batch["text"], return_tensors="pt")
                input_ids = inputs["input_ids"].to(self.device)
                output_dict = self.text_to_mel_model(input_ids, return_dict=True)
                batch["gt_spec"] = output_dict["spectrogram"].transpose(1, 2)
            else:
                sequences, lengths = self.taco_utils.prepare_input_sequence(
                    [batch["text"]]
                )
                with torch.no_grad():
                    mel, _, _ = self.tacotron2.infer(sequences, lengths)
                batch["gt_spec"] = mel
        else:
            assert "get_spectrogram" in self.config

            batch["gt_spec"] = torch.log(self.get_spec(batch["gt_audio"]) + 1e-5)

        outputs = self.generator(**batch)
        batch.update(outputs)

        # Some saving logic. This is an example
        # Use if you need to save predictions on disk
        batch_size = 1
        current_id = batch_idx * batch_size
        self.writer.set_step(current_id)

        for i in range(batch_size):
            # clone because of
            # https://github.com/pytorch/pytorch/issues/1995
            if self.resynthesize:
                ground_truth = batch["gt_audio"][i].clone()
            predict = batch["pr_audio"][i].clone()

            if self.resynthesize:
                torchaudio.save(
                    self.save_path / "gt_audio" / str(batch["filename"] + ".wav"),
                    ground_truth.unsqueeze(0).cpu(),
                    22050,
                    channels_first=True,
                )
                self.log_audio(ground_truth.unsqueeze(0), "gt_audio")

            torchaudio.save(
                self.save_path / "pr_audio" / str(batch["filename"] + ".wav"),
                predict.cpu(),
                22050,
                channels_first=True,
            )
            self.log_audio(predict, "pr_audio")

            self.log_spectrogram(batch["gt_spec"], "gt_spec")
            self.log_spectrogram(batch["pr_spec"], "pr_spec")

        return batch

    def _from_pretrained(self, pretrained_path):
        """
        Init model with weights from pretrained pth file.

        Notice that 'pretrained_path' can be any path on the disk. It is not
        necessary to locate it in the experiment saved dir. The function
        initializes only the model.

        Args:
            pretrained_path (str): path to the model state dict.
        """
        pretrained_path = str(pretrained_path)
        if hasattr(self, "logger"):  # to support both trainer and inferencer
            self.logger.info(f"Loading model weights from: {pretrained_path} ...")
        else:
            print(f"Loading model weights from: {pretrained_path} ...")
        checkpoint = torch.load(pretrained_path, self.device)

        if checkpoint.get("state_dict_g") is not None:
            self.generator.load_state_dict(checkpoint["state_dict_g"])
        else:
            raise ValueError("state_dict should be in the checkpoint")

    def log_audio(self, audio, name):
        if self.writer is not None:
            self.writer.add_audio(name, audio[0], 22050)

    def log_spectrogram(self, spec, name):
        if self.writer is not None:
            spectrogram_for_plot = spec[0].detach().cpu()
            image = plot_spectrogram(spectrogram_for_plot)
            self.writer.add_image(name, image)
