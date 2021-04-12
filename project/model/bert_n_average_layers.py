from .bert_module import BertBaseCaseModel
import hydra
from transformers import AutoModel
from .base_model import BaseModel
import torch


class AverageNLayerBertModel(BertBaseCaseModel):
    def __init__(self, optim, loss, model_name='bert-base-uncased', n_average_layers=4, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.n_average_layers = n_average_layers
        self.model = AutoModel.from_pretrained(model_name,  output_hidden_states=True)
        self.loss_func, self.mining_func = self.get_loss_funcs()

    def forward(self, image, input_ids, attention_mask=None):
        hidden_layers = self.model(input_ids=input_ids, attention_mask=attention_mask)[2]
        text_embedding = self.average_n_layers(hidden_layers)
        return None, None, text_embedding

    def average_n_layers(self, hidden_states):
        hidden_layers = torch.stack(hidden_states, dim=0)
        hidden_layers = hidden_layers.permute(1, 0, 2, 3)
        n_last_layers = hidden_layers[:, -self.n_average_layers:, 0, :]
        cls_embedding = torch.mean(n_last_layers, dim=1)
        return cls_embedding

    def _step(self, batch):
        images_batch, title_ids, attention_masks, label_group = self.extract_input(batch)
        image_text_embedding, image_embedding, text_embedding = self(images_batch, title_ids, attention_masks)
        return image_text_embedding, image_embedding, text_embedding, label_group