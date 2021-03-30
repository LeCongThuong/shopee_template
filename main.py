import logging

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
import torch
logger = logging.getLogger(__name__)


@hydra.main(config_path="configs", config_name="defaults")
def main(cfg: DictConfig) -> None:

    pl.seed_everything(cfg.general.seed)
    device = torch.device("cuda")
    logger.info("\n" + OmegaConf.to_yaml(cfg))

    model_module = hydra.utils.instantiate(cfg.model, _recursive_=False)

    train_class = hydra.utils.instantiate(cfg.data.train)

    val_class = hydra.utils.instantiate(cfg.data.val)

    test_class = hydra.utils.instantiate(cfg.data.test)

    # cfg.data.sampler.train.sampler.data_source = train_class.dataset
    # cfg.data.sampler.val.sampler.data_source = val_class.dataset
    # cfg.data.sampler.test.sampler.data_source = test_class.dataset
    #
    # train_sampler = hydra.utils.instantiate(cfg.data.sampler.train)
    # val_sampler = hydra.utils.instantiate(cfg.data.sampler.val)
    # test_sampler = hydra.utils.instantiate(cfg.data.sampler.test)
    train_sampler = train_class.get_sampler('random', train_class.dataset)
    val_sampler = val_class.get_sampler('sequence', val_class.dataset)
    test_sampler = test_class.get_sampler('sequence', test_class.dataset)

    train_loader = train_class.get_dataloader(train_sampler)
    val_loader = train_class.get_dataloader(val_sampler)
    test_loader = train_class.get_dataloader(test_sampler)

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
    model_module.evaluate_train_dataset(val_loader, cfg.data.val.dataset.csv_annotation_file, cfg.general.topk, device)

    # for restore continue training
    # trainer = Trainer(resume_from_checkpoint='some/path/to/my_checkpoint.ckpt')
    # model_module.predict_test_dataset(test_loader, test_csv_path, output_file_path, topk, device)
if __name__ == "__main__":
    main()





