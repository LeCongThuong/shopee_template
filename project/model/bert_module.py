import hydra
from transformers import AutoModel
from .base_model import BaseModel


class BertBaseCaseModel(BaseModel):
    def __init__(self, model_name='bert-base-uncased', **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.model = AutoModel.from_pretrained(model_name)
        self.loss_func, self.mining_func = self.get_loss_funcs()

    def forward(self, image, input_ids, attention_mask=None):
        return None, None, self.model(input_ids=input_ids, attention_mask=attention_mask)[1]

    def training_step(self,  batch, batch_idx, **kargs):
        image_text_embeddings, image_embeddings, text_embeddings, label_group = self._step(batch)
        text_indices_tuple = self.mining_func(text_embeddings, label_group)
        loss = self.loss_func(text_embeddings, label_group, text_indices_tuple)
        self.log("loss/train", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        output = {"loss": loss}
        return output

    def _step(self, batch):
        images_batch, title_ids, attention_masks, label_group = self.extract_input(batch)
        image_text_embedding, image_embedding, text_embedding = self(images_batch, title_ids, attention_masks)
        return image_text_embedding, image_embedding, text_embedding, label_group

    def get_loss_funcs(self):
        loss_func = hydra.utils.instantiate(self.hparams.loss.loss_func)
        mining_func = hydra.utils.instantiate(self.hparams.loss.mining_func)
        return loss_func, mining_func

    def configure_optimizers(self):
        optim = hydra.utils.instantiate(self.hparams.optim, self.parameters())
        return optim

    def extract_input(self, batch):
        images = batch["images"]
        title_ids = batch["title_ids"]
        label_groups = batch.get("label_groups", None)
        attention_masks = batch["attention_masks"]
        return images, title_ids, attention_masks, label_groups

    def load_model(self, checkpoint_path):
        model = BertBaseCaseModel.load_from_checkpoint(checkpoint_path)
        return model