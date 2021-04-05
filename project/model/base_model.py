import pytorch_lightning as pl
import torch
from project.utils import read_csv
import numpy as np
import pandas as pd
import faiss
from tqdm import tqdm


class BaseModel(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()

    def forward(self, **kwargs):
        raise NotImplementedError

    def training_step(self, **kwargs):
        raise NotImplementedError

    def configure_optimizers(self, **kwargs):
        raise NotImplementedError

    def evaluate_train_dataset(self, val_dataloader, csv_file, threshold, device):
        self.to(device)
        embedding_list = self.get_all_embeddings(val_dataloader, device)
        posting_id_list, target_list = self.process_csv_file(csv_file, test_mode=False)
        k_post_neighbor_pred_list = self.get_k_neighbors(embedding_list, posting_id_list, threshold=threshold)
        f1_score = self.get_f1_dice_score(target_list, k_post_neighbor_pred_list)
        self.log('val/f1_score', f1_score)
        print("Validation: f1_score: ", f1_score)
        with open('f_1_score.txt', 'w') as f:
            f.write(str(f1_score))
        return f1_score

    def get_all_embeddings(self, dataloader, device):
        embedding_list = []
        self.eval()
        for batch_idx, batch in enumerate(tqdm(dataloader, total=int(len(dataloader)))):
            with torch.no_grad():
                images_batch, title_ids, attention_masks, _ = self.extract_input(batch)
                images_batch, title_ids, attention_masks = images_batch.to(device=device), title_ids.to(device=device), attention_masks.to(device)
                image_text_embedding, image_embedding, text_embedding = self(images_batch, title_ids, attention_masks)
            embedding_list.append(image_text_embedding)
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

    def predict_test_dataset(self, test_dataloader, csv_file, output_file_path, threshold, device) -> None:
        self.to(device)
        embedding_list = self.get_all_embeddings(test_dataloader, device)
        posting_id_list, _ = self.process_csv_file(csv_file, test_mode=True)
        k_post_neighbor_pred_list = self.get_k_neighbors(embedding_list, posting_id_list, threshold=threshold)
        self.write_to_csv(posting_id_list, k_post_neighbor_pred_list, output_file_path)

    def write_to_csv(self, posting_id_list, k_post_neighbor_pred_list, output_file_path):
        k_post_neighbor_pred_list = [' '.join(k_post_neighbor_pred) for k_post_neighbor_pred in k_post_neighbor_pred_list]
        submit_dict = {"posting_id": posting_id_list, "matches": k_post_neighbor_pred_list}
        df = pd.DataFrame(submit_dict, columns=['posting_id', 'matches'])
        df.to_csv(output_file_path, index=False)

    def get_k_neighbors(self, embeddings, posting_ids_list, threshold=0.8):
        embeddings = torch.cat(embeddings)
        embeddings = embeddings.detach().cpu().numpy()
        embedding_num, D = embeddings.shape
        cpu_index = faiss.IndexFlatIP(D)
        gpu_index = faiss.index_cpu_to_all_gpus(cpu_index)
        faiss.normalize_L2(embeddings)
        gpu_index.add(embeddings)
        dists, ids = gpu_index.search(embeddings, 50)
        boolean_k_neighbors = dists > threshold
        k_neighbor_list = []
        for i in range(embedding_num):
            k_neighbor_position = ids[i][boolean_k_neighbors[i]]
            k_neighbor_list.append(np.take(posting_ids_list, k_neighbor_position))
        k_neighbor_pred = np.array(k_neighbor_list)
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

