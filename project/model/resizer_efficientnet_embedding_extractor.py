from .efficientnet_embedding_extractor import EfficientNetImageEmbedding
import torch
import timm
from .base_model import BaseModel
import torch.nn as nn
import hydra
from .resizer_module import ResizingNetwork


class ResizerEfficientnetEmbeddingExtractor(EfficientNetImageEmbedding):
    def __init__(self, optim, loss, arch='efficientnet_b3', out_feature=512, dropout_ratio=0.1, r=1, n=16, image_size=300, **kwargs):
        super().__init__(optim, loss, arch, out_feature, dropout_ratio, **kwargs)
        self.save_hyperparameters()
        self.resizer = ResizingNetwork(r, n, image_size)

    def forward(self, image,
                title_ids,
                attention_mask=None,
                ):
        out = self.resizer(image)
        out = self.base_model.forward_features(out)
        out = self.embedding_extractor(out)
        return out, None, None

    def configure_optimizers(self):
        base_optim = hydra.utils.instantiate(self.hparams.optim.base, self.base_model.parameters())
        in_out_params = list(self.embedding_extractor.parameters()) + list(self.resizer.parameters())
        head_optim = hydra.utils.instantiate(self.hparams.optim.head, in_out_params)
        return base_optim, head_optim