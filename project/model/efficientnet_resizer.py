from .efficientnet import EfficientnetExtractor
from .resizer_module import ResizingNetwork


class ResizeEfficientnet(EfficientnetExtractor):
    def __init__(self, optim, loss, arch='efficientnet-b3', r=1, n=16, image_size=224, **kwargs):
        super().__init__(optim, loss, arch, **kwargs)
        self.save_hyperparameters()
        self.resizer_module = ResizingNetwork(r, n, image_size)

    def forward(self, image,
                title_ids,
                attention_mask=None,
                ):
        out = self.resizer_module(image)
        out = self.base.extract_features(out)
        out = self.adaptive_pooling(out)
        out = self.flatten(out)
        return out, None, None