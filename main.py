import logging

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
import torch
from project.data import ShopeeDatasetLoader

logger = logging.getLogger(__name__)


@hydra.main(config_path="configs", config_name="defaults")
def main(cfg: DictConfig) -> None:

    pl.seed_everything(cfg.general.seed)
    device = torch.device("cuda")
    logger.info("\n" + OmegaConf.to_yaml(cfg))

    model_module = hydra.utils.instantiate(cfg.model, _recursive_=False)

    train_loader = hydra.utils.instantiate(cfg.data_loader.train).get_dataloader()
    val_loader = hydra.utils.instantiate(cfg.data_loader.val).get_dataloader()
    # test_loader = hydra.utils.instantiate(cfg.data_loader.test).get_dataloader()

    tensorboard = hydra.utils.instantiate(cfg.logging)

    callbacks = [
        pl.callbacks.ModelCheckpoint(monitor='loss/train_step', dirpath='my/path',
                                     filename='{epoch}-{step}-{loss/train_step:.2f}',
                                     mode='min',
                                     period=2,
                                     save_last=True),
        pl.callbacks.EarlyStopping(monitor='loss/train_step', patience=50),
    ]

    trainer = pl.Trainer(
        logger=tensorboard,
        callbacks=callbacks,
        **cfg.trainer
    )

    model_module.to(device)
    trainer.fit(model_module, train_dataloader=train_loader)
    model_module.evaluate_train_dataset(val_loader, cfg.data_dataset.val.csv_file, cfg.general.topk, device)

    # for restore continue training
    # trainer = Trainer(resume_from_checkpoint='some/path/to/my_checkpoint.ckpt')
    # model_module.predict_test_dataset(test_loader, test_csv_path, output_file_path, topk, device)
if __name__ == "__main__":
    main()





