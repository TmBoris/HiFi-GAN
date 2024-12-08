import logging
import warnings

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.datasets.CustomDirDataset import CustomAudioDirDataset, CustomTextDirDataset
from src.trainer import Inferencer
from src.utils.init_utils import set_random_seed, setup_saving_and_logging
from src.utils.io_utils import ROOT_PATH

warnings.filterwarnings("ignore", category=UserWarning)


@hydra.main(version_base=None, config_path="src/configs", config_name="synthesize")
def main(config):
    """
    Main script for inference. Instantiates the model, metrics, and
    dataloaders. Runs Inferencer to calculate metrics and (or)
    save predictions.

    Args:
        config (DictConfig): hydra experiment config.
    """
    set_random_seed(config.inferencer.seed)

    if config.writer is not None:
        project_config = OmegaConf.to_container(config)
        logger = logging.getLogger("synth")
        logger.setLevel(logging.DEBUG)
        writer = instantiate(config.writer, logger, project_config)
    else:
        writer = None

    if config.inferencer.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.inferencer.device

    if config.inferencer.input_text_dir is not None:
        dataset = CustomTextDirDataset(ROOT_PATH / config.inferencer.input_text_dir)
        samples = dataset.get_texts()
    elif config.inferencer.input_text is not None:
        samples = [{"text": config.inferencer.input_text, "filename": "example"}]
    elif config.inferencer.input_audio_dir is not None:
        dataset = CustomAudioDirDataset(ROOT_PATH / config.inferencer.input_audio_dir)
        samples = dataset.get_audios()
    else:
        raise ValueError(
            "You should give the input in form of text, directory with texts or directory with audios"
        )

    # build model architecture, then print to console
    generator = instantiate(config.generator.model).to(device)
    print(generator)

    # save_path for model predictions
    save_path = ROOT_PATH / "data" / "saved" / config.inferencer.save_path
    save_path.mkdir(exist_ok=True, parents=True)

    inferencer = Inferencer(
        generator=generator,
        config=config,
        device=device,
        samples=samples,
        save_path=save_path,
        writer=writer,
        text_to_mel_model=config.inferencer.text_to_mel_model,
        resynthesize=(config.inferencer.input_audio_dir is not None),
        skip_model_load=False,
    )

    inferencer.run_inference()


if __name__ == "__main__":
    main()
