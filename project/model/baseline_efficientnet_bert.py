import pytorch_lightning as pl
import torch
import hydra
from efficientnet_pytorch import EfficientNet
import torch.nn as nn
from transformers import AutoModel
from project.utils import read_csv
import numpy as np
import pandas as pd
import faiss
from tqdm import tqdm


class EfficientnetModel(pl.LightningModule):
    def __init__(self, arch='efficientnet-b0', out_feature=512, dropout_ratio=0.2, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.base = EfficientNet.from_pretrained(arch)
        self.adaptive_pooling = nn.AdaptiveAvgPool2d(output_size=1)
        self.drop_out = nn.Dropout(p=dropout_ratio, inplace=False)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(self.base._fc.in_features, out_feature)

    def forward(self, x):
        out = self.base.extract_features(x)
        out = self.adaptive_pooling(out)
        out = self.drop_out(out)
        out = self.flatten(out)
        out = self.fc(out)
        return out


class BertBaseCaseModel(pl.LightningModule):
    def __init__(self, model_name='bert-base-uncased'):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)

    def forward(self, input_ids, attention_mask=None):
        return self.model(input_ids=input_ids, attention_mask=attention_mask)[1]


class BaselineModel(pl.LightningModule):
    def __init__(self, optim, loss, arch='efficientnet-b0', out_feature=512, dropout_ratio=0.2,  model_name='bert-base-cased', **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.image_extractor = EfficientnetModel(arch=arch, out_feature=out_feature, dropout_ratio=dropout_ratio)
        self.text_extractor = BertBaseCaseModel(model_name=model_name)
        self.loss_func, self.mining_func = self.get_loss_funcs()

    def forward(self, image,
                title_ids,
                attention_mask=None,
                ):
        image_embedding = self.image_extractor(image)
        text_embedding = self.text_extractor(title_ids, attention_mask)
        return torch.cat([image_embedding, text_embedding], dim=1)

    def _step(self, batch):
        images_batch, title_ids, attention_masks, label_group = self.extract_input(batch)
        embedding = self(images_batch, title_ids, attention_masks)
        return embedding, label_group

    def training_step(self, batch, batch_idx):
        embeddings, label_group = self._step(batch)
        indices_tuple = self.mining_func(embeddings, label_group)
        loss = self.loss_func(embeddings, label_group, indices_tuple)
        self.log("loss/train", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)

    def get_loss_funcs(self):
        loss_func = hydra.utils.instantiate(self.hparams.loss.loss_func)
        mining_func = hydra.utils.instantiate(self.hparams.loss.mining_func)
        return loss_func, mining_func

    def evaluate_train_dataset(self, val_dataloader, csv_file, topk, device):
        embedding_list = self.get_all_embeddings(val_dataloader, device)
        posting_id_list, target_list = self.process_csv_file(csv_file, test_mode=False)
        k_post_neighbor_pred_list = self.get_k_neighbors(embedding_list, posting_id_list, topk=topk)
        f1_score = self.get_f1_dice_score(target_list, k_post_neighbor_pred_list)
        self.log('val/f1_score', f1_score)
        print("Validation: f1_score: ", f1_score)
        return f1_score

    def get_all_embeddings(self, dataloader, device):
        embedding_list = []
        self.eval()
        for batch_idx, batch in enumerate(tqdm(dataloader, total=int(len(dataloader)))):
            with torch.no_grad():
                images_batch, title_ids, attention_masks, _ = self.extract_input(batch)
                images_batch, title_ids, attention_masks = images_batch.to(device=device), title_ids.to(device=device), attention_masks.to(device)
                embedding = self(images_batch, title_ids, attention_masks)
            embedding_list.append(embedding)
        return embedding_list

    def process_csv_file(self, csv_file, test_mode=False):
        df = read_csv(csv_file)
        posting_id_list = df['posting_id'].to_numpy()
        if test_mode:
            return posting_id_list, None
        if 'target' not in df.columns:
            target_dict = df.groupby('label_group').posting_id.agg('unique').to_dict()
            df['target'] = df.label_group.map(target_dict)
        target_list = df['target'].to_numpy()
        return posting_id_list, target_list

    def predict_test_dataset(self, test_dataloader, csv_file, output_file_path, topk, device) -> None:
        embedding_list = self.get_all_embeddings(test_dataloader, device)
        posting_id_list, _ = self.process_csv_file(csv_file, test_mode=True)
        k_post_neighbor_pred_list = self.get_k_neighbors(embedding_list, posting_id_list, topk=topk)
        self.write_to_csv(posting_id_list, k_post_neighbor_pred_list, output_file_path)

    def write_to_csv(self, posting_id_list, k_post_neighbor_pred_list, output_file_path):
        k_post_neighbor_pred_list = [' '.join(k_post_neighbor_pred) for k_post_neighbor_pred in k_post_neighbor_pred_list]
        submit_dict = {"posting_id": posting_id_list, "matches": k_post_neighbor_pred_list}
        df = pd.DataFrame(submit_dict, columns=['posting_id', 'matches'])
        df.to_csv(output_file_path, index=False)

    def get_k_neighbors(self, embeddings, posting_ids_list, topk=50):
        embeddings = torch.cat(embeddings)
        embeddings = embeddings.detach().cpu().numpy()
        embedding_num, D = embeddings.shape
        cpu_index = faiss.IndexFlatL2(D)
        gpu_index = faiss.index_cpu_to_all_gpus(cpu_index)
        gpu_index.add(embeddings)
        dists, ids = gpu_index.search(x=embeddings, k=topk)
        k_neighbor_list = []
        for i in range(embedding_num):
            k_neighbor_list.append(np.take(posting_ids_list, ids[i, :]))
        k_neighbor_pred = np.vstack(k_neighbor_list)
        return k_neighbor_pred

    def get_f1_dice_score(self, targets, neighbor_preds):
        length_num = targets.shape[0]
        f1_score = []
        for i in range(length_num):
            neighbor_pred = neighbor_preds[i]
            target = targets[i]
            f1_score_row = self.calculate_f1_dice(neighbor_pred, target)
            f1_score.append(f1_score_row)
        f1_score = np.array(f1_score).mean()
        return f1_score

    def calculate_f1_dice(self, neighbor_pred, target):
        n = len(np.intersect1d(neighbor_pred, target))
        return 2 * n / (len(neighbor_pred) + len(target))

    def configure_optimizers(self):
        return hydra.utils.instantiate(self.hparams.optim, self.parameters())

    def extract_input(self, batch):
        images = batch["images"]
        title_ids = batch["title_ids"]
        label_groups = batch.get("label_groups", None)
        attention_masks = batch["attention_masks"]
        return images, title_ids, attention_masks, label_groups

    def load_model(self, checkpoint_path):
        model = BaselineModel.load_from_checkpoint(checkpoint_path)
        return model

