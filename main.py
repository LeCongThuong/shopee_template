import logging

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from project.model.baseline_efficientnet_bert import BaselineModel
from project.data.adaptive_resizer_batch import ShopeeDatasetLoader, AdaptiveResizerDataset
from project.augmentation.image_albumentation import AlbumentationAugment
from project.text_preprocess.text_preprocess import  TextProcessing
import torch
logger = logging.getLogger(__name__)


@hydra.main(config_path="configs", config_name="defaults")
def main(cfg: DictConfig) -> None:
    pl.seed_everything(1234)
    logger.info("\n" + OmegaConf.to_yaml(cfg))

    arch = 'efficientnet-b0'
    out_feature = 512
    dropout_ratio = 0.2
    tokenizer_str = 'bert-base-cased'
    model_name = 'bert-base-cased'
    image_dir = 'shopee'
    train_csv_path = '.csv'
    test_csv_path = '.csv'
    text_max_length = 16
    text_padding = 'longest'
    is_truncate = True
    output_file_path = '/content/submission.csv'
    batch_size = 8
    sampler_str = 'random'
    topk=50
    device = torch.device("cuda")

    transform = AlbumentationAugment()
    text_preprocess = TextProcessing()
    model_module = BaselineModel(arch, out_feature, dropout_ratio, tokenizer_str, model_name)

    train_dataset = AdaptiveResizerDataset(image_dir, train_csv_path, transform, text_preprocess, model_module.tokenizer,
                                           text_max_length, text_padding=text_padding, is_truncate=is_truncate, do_train=True)
    val_dataset = AdaptiveResizerDataset(image_dir, train_csv_path, None, text_preprocess, model_module.tokenizer, text_max_length,
                                         text_padding='longest', is_truncate=True, do_train=True)
    train_loader = ShopeeDatasetLoader(batch_size, sampler_str, train_dataset).get_dataloder()
    val_loader = ShopeeDatasetLoader(batch_size, 'sequence', val_dataset).get_dataloder()

    # test_dataset = AdaptiveResizerDataset(image_dir, test_csv_path, None, text_preprocess, model_module.tokenizer, text_max_length,
    #                                      text_padding='longest', is_truncate=True, do_train=False)
    # test_loader = ShopeeDatasetLoader(batch_size, 'sequence', test_dataset).get_dataloder()

    tensorboard = pl.loggers.TensorBoardLogger(".", "default", "", log_graph=True, default_hp_metric=False)

    gpus = 1
    max_epochs = 10
    progress_bar_refresh_rate = 30
    log_every_n_steps = 50
    num_sanity_val_steps = 0

    trainer = pl.Trainer(
        gpus=gpus,
        logger=tensorboard,
        callbacks=model_module.get_callbacks(),
        log_every_n_steps=log_every_n_steps,
        max_epochs=max_epochs,
        num_sanity_val_steps=num_sanity_val_steps,
        progress_bar_refresh_rate=progress_bar_refresh_rate
    )

    model_module.to(device)
    trainer.fit(model_module, train_dataloader=train_loader)
    model_module.evaluate_train_dataset(val_loader, train_csv_path, topk, device)

    # for restore continue training
    # trainer = Trainer(resume_from_checkpoint='some/path/to/my_checkpoint.ckpt')
    # model_module.predict_test_dataset(test_loader, test_csv_path, output_file_path, topk, device)






