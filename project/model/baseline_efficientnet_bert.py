import pytorch_lightning as pl
import torch
import hydra
from efficientnet_pytorch import EfficientNet
import torch.nn as nn
from transformers import AutoModel
from .base_model import BaseModel


# English --> 18939 samples, Indonesia -->8715 samples, Malay --> 2398 samples, German --> 854 samples
class EfficientnetModel(pl.LightningModule):
    def __init__(self, arch='efficientnet-b0', out_feature=768, dropout_ratio=0.2, **kwargs):
        super().__init__()
        self.base = EfficientNet.from_pretrained(arch)
        self.adaptive_pooling = nn.AdaptiveAvgPool2d(output_size=1)
        self.drop_out = nn.Dropout(p=dropout_ratio, inplace=False)
        self.flatten = nn.Flatten()
        # self.fc = nn.Linear(self.base._fc.in_features, out_feature)

    def forward(self, x):
        out = self.base.extract_features(x)
        out = self.adaptive_pooling(out)
        out = self.drop_out(out)
        out = self.flatten(out)
        # out = self.fc(out)
        return out


class Self_Attn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim):
        super(Self_Attn, self).__init__()
        self.chanel_in = in_dim
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)  #

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x
        return out


class BertBaseCaseModel(pl.LightningModule):
    def __init__(self, model_name='bert-base-uncased'):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)

    def forward(self, input_ids, attention_mask=None):
        return self.model(input_ids=input_ids, attention_mask=attention_mask)[1]


class BaselineModel(BaseModel):
    def __init__(self, optim, loss, arch='efficientnet-b0', out_feature=512, dropout_ratio=0.2,  model_name='bert-base-cased', **kwargs):
        super().__init__()
        self.image_extractor = EfficientnetModel(arch=arch, out_feature=out_feature, dropout_ratio=dropout_ratio)
        self.text_extractor = BertBaseCaseModel(model_name=model_name)
        self.loss_func, self.mining_func = self.get_loss_funcs()
        self.self_attention = Self_Attn(36)
        self.linear = torch.nn.Linear(768+1536, out_feature)
        self.flatten = nn.Flatten()

    def forward(self, image,
                title_ids,
                attention_mask=None,
                ):
        image_embedding = self.image_extractor(image)
        text_embedding = self.text_extractor(title_ids, attention_mask)
        image_text_concat = torch.cat([image_embedding, text_embedding], dim=1)
        image_text_concat = image_text_concat.view(-1, 36, 8, 8)
        image_text_embedding = self.self_attention(image_text_concat)
        image_text_embedding = self.flatten(image_text_embedding)
        image_text_embedding = self.linear(image_text_embedding)
        return image_text_embedding, image_embedding, text_embedding

    def _step(self, batch):
        images_batch, title_ids, attention_masks, label_group = self.extract_input(batch)
        image_text_embedding, image_embedding, text_embedding = self(images_batch, title_ids, attention_masks)
        return image_text_embedding, image_embedding, text_embedding, label_group

    def training_step(self, batch, batch_idx, optimizer_idx):
        image_text_embeddings, image_embeddings, text_embeddings, label_group = self._step(batch)
        image_text_indices_tuple = self.mining_func(image_text_embeddings, label_group)
        image_indices_tuple = self.mining_func(image_embeddings, label_group)
        text_indices_tuple = self.mining_func(text_embeddings, label_group)
        image_text_loss = self.loss_func(image_text_embeddings, label_group, image_text_indices_tuple)
        image_loss = self.loss_func(image_embeddings, label_group, image_indices_tuple)
        text_loss = self.loss_func(text_embeddings, label_group, text_indices_tuple)
        loss = image_text_loss + image_loss + text_loss
        self.log("loss/train_image_text", image_text_loss, prog_bar=True, logger=True, on_step=True, on_epoch=False)
        self.log("loss/train_image", image_loss, prog_bar=True, logger=True, on_step=True, on_epoch=False)
        self.log("loss/train_text", text_loss, prog_bar=True, logger=True, on_step=True, on_epoch=False)
        self.log("loss/train", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        output = {"loss": loss}
        return output

    def get_loss_funcs(self):
        loss_func = hydra.utils.instantiate(self.hparams.loss.loss_func)
        mining_func = hydra.utils.instantiate(self.hparams.loss.mining_func)
        return loss_func, mining_func

    def configure_optimizers(self):
        text_extractor_optim = hydra.utils.instantiate(self.hparams.optim.backbone, self.text_extractor.parameters())
        image_extractor_optim = hydra.utils.instantiate(self.hparams.optim.backbone, self.image_extractor.parameters())
        self_attention_optim = hydra.utils.instantiate(self.hparams.optim.head, self.self_attention.parameters())
        linear_optim = hydra.utils.instantiate(self.hparams.optim.head, self.linear.parameters())
        return [text_extractor_optim, image_extractor_optim, self_attention_optim, linear_optim]

    def extract_input(self, batch):
        images = batch["images"]
        title_ids = batch["title_ids"]
        label_groups = batch.get("label_groups", None)
        attention_masks = batch["attention_masks"]
        return images, title_ids, attention_masks, label_groups

    def load_model(self, checkpoint_path):
        model = BaselineModel.load_from_checkpoint(checkpoint_path)
        return model
