import torch
import timm
from .base_model import BaseModel
import torch.nn as nn
import hydra


class EfficientNetImageEmbedding(BaseModel):
    def __init__(self, optim, loss, arch='efficientnet_b3', out_feature=512, dropout_ratio=0.1, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.base_model = timm.create_model(arch, pretrained=True)
        self.embedding_extractor = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Dropout(p=dropout_ratio, inplace=False),
            nn.Flatten(),
            nn.Linear(self.base_model.classifier.in_features, out_feature),
        )
        self.loss_func, self.mining_func = self.get_loss_funcs()

    def forward(self, image,
                title_ids,
                attention_mask=None,
                ):
        out = self.base_model.forward_features(image)
        out = self.embedding_extractor(out)
        return None, out, None

    def _step(self, batch):
        images_batch, title_ids, attention_masks, label_group = self.extract_input(batch)
        image_text_embedding, image_embedding, text_embedding = self(images_batch, title_ids, attention_masks)
        return image_text_embedding, image_embedding, text_embedding, label_group

    def training_step(self, batch, batch_idx,optimizer_idx, **kargs):
        image_text_embeddings, image_embeddings, text_embeddings, label_group = self._step(batch)
        image_indices_tuple = self.mining_func(image_embeddings, label_group)
        loss = self.loss_func(image_text_embeddings, label_group, image_indices_tuple)
        self.log("loss/train", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        output = {"loss": loss}
        return output

    def extract_input(self, batch):
        images = batch["images"]
        title_ids = batch["title_ids"]
        label_groups = batch.get("label_groups", None)
        attention_masks = batch["attention_masks"]
        return images, title_ids, attention_masks, label_groups

    def get_loss_funcs(self):
        loss_func = hydra.utils.instantiate(self.hparams.loss.loss_func)
        mining_func = hydra.utils.instantiate(self.hparams.loss.mining_func)
        return loss_func, mining_func

    def configure_optimizers(self):
        base_optim = hydra.utils.instantiate(self.hparams.optim.base, self.base_model.parameters())
        head_optim = hydra.utils.instantiate(self.hparams.optim.head, self.embedding_extractor.parameters())
        return base_optim, head_optim

    def load_model(self, checkpoint_path):
        model = EfficientNetImageEmbedding.load_from_checkpoint(checkpoint_path)
        return model
