import warnings

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.datasets.data_utils import get_dataloaders
from src.trainer import Trainer
from src.utils.init_utils import set_random_seed, setup_saving_and_logging

warnings.filterwarnings("ignore", category=UserWarning)


@hydra.main(version_base=None, config_path="src/configs", config_name="hifigan")
def main(config):
    """
    Main script for training. Instantiates the model, optimizer, scheduler,
    metrics, logger, writer, and dataloaders. Runs Trainer to train and
    evaluate the model.

    Args:
        config (DictConfig): hydra experiment config.
    """
    set_random_seed(config.trainer.seed)

    project_config = OmegaConf.to_container(config)
    logger = setup_saving_and_logging(config)
    writer = instantiate(config.writer, logger, project_config)

    if config.trainer.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.trainer.device

    # setup data_loader instances
    # batch_transforms should be put on device
    dataloaders, batch_transforms = get_dataloaders(config, device)
    
    # build model architecture, then print to console
    generator = instantiate(config.generator.model).to(device)
    discriminator = instantiate(config.discriminator.model).to(device)
    logger.info(generator)
    logger.info(discriminator)

    # get function handles of loss and metrics
    gen_loss = instantiate(config.gen_loss).to(device)
    discr_loss = instantiate(config.discr_loss).to(device)

    metrics = instantiate(config.metrics)

    # build generator, learning rate scheduler
    g_params = filter(lambda p: p.requires_grad, generator.parameters())
    optim_g = instantiate(config.optim_g, params=g_params)
    lr_sched_g = instantiate(config.lr_sched_g, optimizer=optim_g)

    d_params = filter(lambda p: p.requires_grad, discriminator.parameters())
    optim_d = instantiate(config.optim_d, params=d_params)
    lr_sched_d = instantiate(config.lr_sched_d, optimizer=optim_d)

    # epoch_len = number of iterations for iteration-based training
    # epoch_len = None or len(dataloader) for epoch-based training
    epoch_len = config.trainer.get("epoch_len")

    trainer = Trainer(
        generator=generator,
        discriminator=discriminator,
        gen_loss=gen_loss,
        discr_loss=discr_loss,
        optim_g=optim_g,
        optim_d=optim_d,
        lr_sched_g=lr_sched_g,
        lr_sched_d=lr_sched_d,
        metrics=metrics,
        config=config,
        device=device,
        dataloaders=dataloaders,
        epoch_len=epoch_len,
        logger=logger,
        writer=writer,
        batch_transforms=batch_transforms,
        skip_oom=config.trainer.get("skip_oom", True),
    )

    trainer.train()


if __name__ == "__main__":
    main()
