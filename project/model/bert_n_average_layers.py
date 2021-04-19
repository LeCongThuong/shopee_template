from .bert_module import BertBaseCaseModel
import hydra
from transformers import AutoModel
from .base_model import BaseModel
import torch
import pytorch_lightning as pl


class AverageVetorModule(pl.LightningModule):
    def __init__(self, n_layers):
        super(AverageVetorModule, self).__init__()
        self.weights = torch.nn.Parameter(torch.randn(n_layers,),  requires_grad=True)
        self.n_layers = n_layers

    def forward(self, embedding):
        for i in range(self.n_layers):
            embedding[:, i, :] = embedding[:, i, :] * self.weights[i]
        return torch.mean(embedding, dim=1)


class AverageNLayerBertModel(BaseModel):
    def __init__(self, model_name='bert-base-uncased', n_average_layers=4, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.n_average_layers = n_average_layers
        self.model = AutoModel.from_pretrained(model_name,  output_hidden_states=True)
        self.model.train()
        self.average_vector = AverageVetorModule(n_average_layers)
        self.loss_func, self.mining_func = self.get_loss_funcs()

    def forward(self, image, input_ids, attention_mask=None):
        hidden_layers = self.model(input_ids=input_ids, attention_mask=attention_mask)[2]
        text_embedding = self.average_n_layers(hidden_layers)
        return None, None, text_embedding

    def average_n_layers(self, hidden_states):
        hidden_layers = torch.stack(hidden_states, dim=0)
        hidden_layers = hidden_layers.permute(1, 0, 2, 3)
        n_last_layers = hidden_layers[:, -self.n_average_layers:, 0, :]
        cls_embedding = self.average_vector(n_last_layers)
        return cls_embedding

    def _step(self, batch):
        images_batch, title_ids, attention_masks, label_group = self.extract_input(batch)
        image_text_embedding, image_embedding, text_embedding = self(images_batch, title_ids, attention_masks)
        return image_text_embedding, image_embedding, text_embedding, label_group

    def get_loss_funcs(self):
        loss_func = hydra.utils.instantiate(self.hparams.loss.loss_func)
        mining_func = hydra.utils.instantiate(self.hparams.loss.mining_func)
        return loss_func, mining_func

    def training_step(self,  batch, batch_idx, **kargs):
        image_text_embeddings, image_embeddings, text_embeddings, label_group = self._step(batch)
        text_indices_tuple = self.mining_func(text_embeddings, label_group)
        loss = self.loss_func(text_embeddings, label_group, text_indices_tuple)
        self.log("loss/train", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        output = {"loss": loss}
        return output

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
        model = AverageNLayerBertModel.load_from_checkpoint(checkpoint_path)
        return model