import logging

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
import torch
from project.data import ShopeeDatasetLoader
import os

logger = logging.getLogger(__name__)


@hydra.main(config_path="configs", config_name="defaults")
def main(cfg: DictConfig) -> None:
    torch.backends.cudnn.benchmark = True
    pl.seed_everything(cfg.general.seed)
    device = torch.device("cuda")
    logger.info("\n" + OmegaConf.to_yaml(cfg))

    model_module = hydra.utils.instantiate(cfg.model, _recursive_=False)

    train_dataset = hydra.utils.instantiate(cfg.data_dataset.train)

    train_sampler = hydra.utils.instantiate(cfg.data_sampler.train,
                                            labels=train_dataset.label_group_list,
                                            length_before_new_iter=len(train_dataset.label_group_list))

    train_loader = hydra.utils.instantiate(cfg.data_loader.train,
                                           dataset=train_dataset,
                                           sampler=train_sampler).get_dataloader()

    val_loader = hydra.utils.instantiate(cfg.data_loader.val).get_dataloader()
    # test_loader = hydra.utils.instantiate(cfg.data_loader.test).get_dataloader()

    tensorboard = hydra.utils.instantiate(cfg.logging)

    # callbacks = [
    #     pl.callbacks.ModelCheckpoint(monitor='loss/train_step',
    #                                  dirpath='checkpoints',
    #                                  filename='{epoch}  -{step}-{loss/train_step:.2f}',
    #                                  mode='min',
    #                                  period=2,
    #                                  save_last=True),
    #     pl.callbacks.EarlyStopping(monitor='loss/train_step', patience=50),
    # ]
    callbacks = [
        hydra.utils.instantiate(cfg.callback.model_checkpoint),
        hydra.utils.instantiate(cfg.callback.early_stop)
    ]

    trainer = pl.Trainer(
        logger=tensorboard,
        callbacks=callbacks,
        **cfg.trainer
    )
    os.makedirs(cfg.general.result_dir, exist_ok=True)
    # lr_finder = trainer.tuner.lr_find(model_module, train_dataloader=train_loader)
    # suggested_lr = lr_finder.suggestion()
    # print("Suggest: ", suggested_lr)
    trainer.fit(model_module, train_dataloader=train_loader)
    model_module.evaluate_train_dataset(val_loader, cfg.data_dataset.val.csv_file, cfg.general.threshold, device)
    # model_module.visual_similar_image_result(val_loader, cfg.data_dataset.val.csv_file, cfg.general.threshold, device, cfg.data_dataset.train.image_dir, cfg.general.result_dir, num_image=50, k_show=6)
    # model_module.load_model(checkpoint_path='/content/shopee_template/outputs/baseline/2021-04-01_101622/checkpoints/last.ckpt')
    # for restore continue training
    # trainer = Trainer(resume_from_checkpoint='some/path/to/my_checkpoint.ckpt')
    # model_module.predict_test_dataset(test_loader, test_csv_path, output_file_path, threshold, device)
if __name__ == "__main__":
    main()





