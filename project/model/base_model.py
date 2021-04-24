import pytorch_lightning as pl
import torch
from project.utils import read_csv
import numpy as np
import pandas as pd
import faiss
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import pickle


class BaseModel(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, **kwargs):
        raise NotImplementedError

    def training_step(self, **kwargs):
        raise NotImplementedError

    def configure_optimizers(self, **kwargs):
        raise NotImplementedError

    def evaluate_train_dataset(self, val_dataloader, csv_file, threshold, device):
        self.to(device)
        embedding_dict = self.get_all_embeddings(val_dataloader, device)
        with open('embedding.plk', 'wb') as f:
            pickle.dump(embedding_dict, f)
        print("Done dump pickle file")
        posting_id_list, target_list = self.process_csv_file(csv_file, test_mode=False)
        k_post_neighbor_pred_list = self.get_k_neighbors(embedding_dict, posting_id_list, threshold=threshold)
        f1_score = self.get_f1_dice_score(target_list, k_post_neighbor_pred_list)
        self.log('val/f1_score', f1_score)
        print("Validation: f1_score: ", f1_score)
        with open('f_1_score.txt', 'w') as f:
            f.write(str(f1_score))
        return f1_score

    def evaluate_train_dataset(self, test_dataloader, csv_file, threshold, max_candidates, device):
        self.to(device)
        embedding_dict = self.get_all_embeddings(test_dataloader, device)
        posting_id_list, target_list = self.process_csv_file(csv_file, test_mode=False)
        with open('embedding.plk', 'wb') as f:
            pickle.dump(embedding_dict, f)
        print("Done dump pickle file")
        predict = self.make_prediction_based_embeddings(embedding_dict, posting_id_list, threshold, max_candidates)
        f1_score = self.get_f1_dice_score(target_list, predict)
        self.log('val/f1_score', f1_score)
        print("Validation: f1_score: ", f1_score)
        with open('f_1_score.txt', 'w') as f:
            f.write(str(f1_score))
        return f1_score

    def make_prediction_based_embeddings(self, embedding_dict, posting_id_list, threshold, max_candidates):
        image_text_embeddings = embedding_dict['image_text_embedding']
        image_embeddings = embedding_dict['image_embedding']
        text_embeddings = embedding_dict['text_embedding']
        image_text_dists, image_text_ids = self.calculate_dist(image_text_embeddings, max_columns=max_candidates)
        image_dists, image_ids = self.calculate_dist(image_embeddings, max_columns=max_candidates)
        text_dists, text_ids = self.calculate_dist(text_embeddings, max_columns=max_candidates)
        predict = self.process_dists(image_text_dists, image_text_ids, image_dists, image_ids, text_dists, text_ids,
                                     threshold, posting_id_list)
        return predict

    def process_dists(self, image_text_dists, image_text_ids, image_dists, image_ids, text_dists, text_ids, threshold, posting_ids_list):
        # {"k_neighbor_id": k_neighbor_pred, "k_dist": k_dist_list}
        k_image_text_pred_core_dict = self.predict_k_most_sim(image_text_dists, image_text_ids, posting_ids_list, threshold=threshold['image_text'][1])
        k_text_pred_core_dict = self.predict_k_most_sim(text_dists, text_ids, posting_ids_list, threshold=threshold['text'][1])
        k_image_pred_core_dict = self.predict_k_most_sim(image_dists, image_ids, posting_ids_list, threshold=threshold['image'][1])

        k_image_text_pred_dict = self.predict_k_most_sim(image_text_dists, image_ids, posting_ids_list, threshold=threshold['image_text'][0])
        k_text_pred_dict = self.predict_k_most_sim(text_dists, image_ids, posting_ids_list, threshold=threshold['text'][0])
        k_image_pred_dict = self.predict_k_most_sim(text_dists, text_ids, posting_ids_list, threshold=threshold['image'][0])

        text_image_ids = np.hstack([k_image_pred_dict["k_neighbor_id"], k_text_pred_dict["k_neighbor_id"]])
        text_2_image_ids = np.hstack([k_image_text_pred_dict["k_neighbor_id"], k_text_pred_dict["k_neighbor_id"]])
        image_2_text_ids = np.hstack([k_image_text_pred_dict["k_neighbor_id"], k_image_pred_dict["k_neighbor_id"]])

        processed_image_text = self.filter_pred(k_image_text_pred_core_dict['k_neighbor_id'], text_image_ids)
        processed_image = self.filter_pred(k_image_pred_core_dict['k_neighbor_id'], text_2_image_ids)
        process_text = self.filter_pred(k_text_pred_core_dict['k_neighbor_id'], image_2_text_ids)
        predict = np.hstack([processed_image_text, processed_image, process_text])
        out = []
        for i in range(predict.shape[0]):
            row = np.array(list(set(predict[i])))
            out.append(row)
        return out

    def filter_pred(self, k_neighbor_core_id, other_preds):
        num_embedding = k_neighbor_core_id.shape[0]
        processed_neighbor_list = []
        for i in range(num_embedding):
            processed_neighbor = []
            k_neighbor_core = k_neighbor_core_id[i]
            other_pred = set(other_preds[i])
            for core_neighbor in k_neighbor_core:
                if core_neighbor in other_pred:
                    processed_neighbor.append(core_neighbor)
            processed_neighbor_list.append(processed_neighbor)
        return np.array(processed_neighbor_list)

    def get_all_embeddings(self, dataloader, device):
        embedding_list = []
        self.eval()
        for batch_idx, batch in enumerate(tqdm(dataloader, total=int(len(dataloader)))):
            with torch.no_grad():
                images_batch, title_ids, attention_masks, _ = self.extract_input(batch)
                images_batch, title_ids, attention_masks = images_batch.to(device=device), title_ids.to(device=device), attention_masks.to(device)
                image_text_embedding, image_embedding, text_embedding = self(images_batch, title_ids, attention_masks)
            embedding_list.append(image_embedding)
        return embedding_list

    def get_all_embeddings(self, dataloader, device):
        image_embedding_list = []
        text_embedding_list = []
        image_text_embedding_list = []
        self.eval()
        for batch_idx, batch in enumerate(tqdm(dataloader, total=int(len(dataloader)))):
            with torch.no_grad():
                images_batch, title_ids, attention_masks, _ = self.extract_input(batch)
                images_batch, title_ids, attention_masks = images_batch.to(device=device), title_ids.to(device=device), attention_masks.to(device)
                image_text_embedding, image_embedding, text_embedding = self(images_batch, title_ids, attention_masks)
                image_embedding_list.append(image_embedding)
                text_embedding_list.append(text_embedding)
                image_text_embedding_list.append(image_text_embedding)
        return {'image_embedding': image_embedding_list, 'text_embedding': text_embedding_list, 'image_text_embedding': image_text_embedding_list}

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

    def process_csv_file_for_visualization(self, csv_file, test_mode=False):
        df = read_csv(csv_file)
        posting_id_list = df['posting_id'].to_numpy()
        image_list = df['image'].to_list()
        image_title = df['title'].to_list()
        if test_mode:
            return posting_id_list, None
        if 'target' not in df.columns:
            target_dict = df.groupby('label_group').posting_id.agg('unique').to_dict()
            df['target'] = df.label_group.map(target_dict)
        target_list = df['target'].to_numpy()
        return posting_id_list, image_list, image_title, target_list

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

    def calculate_dist(self, embeddings, max_columns=50):
        embeddings = torch.cat(embeddings)
        embeddings = embeddings.detach().cpu().numpy()
        embedding_num, D = embeddings.shape
        cpu_index = faiss.IndexFlatIP(D)
        gpu_index = faiss.index_cpu_to_all_gpus(cpu_index)
        faiss.normalize_L2(embeddings)
        gpu_index.add(embeddings)
        dists, ids = gpu_index.search(embeddings, max_columns)
        return dists, ids

    def predict_k_most_sim(self, dists, ids, posting_ids_list, threshold=0.85):
        embedding_num = dists.shape[0]
        boolean_k_neighbors = dists > threshold
        k_neighbor_list = []
        k_dist_list = []
        for i in range(embedding_num):
            k_neighbor_position = ids[i][boolean_k_neighbors[i]]
            k_dist = dists[i][boolean_k_neighbors[i]]
            k_dist_list.append(k_dist)
            k_neighbor_list.append(np.take(posting_ids_list, k_neighbor_position))
        k_neighbor_pred = np.array(k_neighbor_list)
        return {"k_neighbor_id": k_neighbor_pred, "k_dist": k_dist_list}

    def get_k_neighbors(self, embeddings, posting_ids_list, threshold=0.8, return_dist=False):
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
        if return_dist:
            k_dist_list = []
        for i in range(embedding_num):
            k_neighbor_position = ids[i][boolean_k_neighbors[i]]
            if return_dist:
                k_dist = dists[i][boolean_k_neighbors[i]]
                k_dist_list.append(k_dist)
            k_neighbor_list.append(np.take(posting_ids_list, k_neighbor_position))
        k_neighbor_pred = np.array(k_neighbor_list)
        if return_dist:
            return k_neighbor_pred, np.array(k_dist_list)
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

    def visual_similar_image_result(self, dataloader, csv_file, threshold, device, image_source, result_dir, num_image=50, k_show=7):
        self.to(device)
        if os.path.isfile('embedding.plk'):
            with open('embedding.plk', 'rb') as f :
                embedding_list = pickle.load(f)
        else:
            embedding_list = self.get_all_embeddings(dataloader, device)
        posting_id_list, image_list, title_list, target_list = self.process_csv_file_for_visualization(csv_file, test_mode=False)

        post_image_dict = dict(zip(posting_id_list, image_list))
        post_title_dict = dict(zip(posting_id_list, title_list))

        k_post_neighbor_pred_list, k_dist_list = self.get_k_neighbors(embedding_list, posting_id_list, threshold=threshold, return_dist=True)
        chosen_postion_list = np.random.choice(len(posting_id_list), num_image, replace=False)
        chosen_posting_id_list = posting_id_list[chosen_postion_list]
        # chosen_image_list = []
        # for position in chosen_postion_list:
        #     chosen_image_list.append(os.path.join(image_source, image_list[position]))
        # chosen_image_list = [os.path.join(image_source, image_name) for image_name in image_list[chosen_postion_list]]
        chosen_target_list = target_list[chosen_postion_list]
        chosen_pred_list = k_post_neighbor_pred_list[chosen_postion_list]
        chosen_k_distance = k_dist_list[chosen_postion_list]

        for i in range(num_image):
            chosen_post = chosen_posting_id_list[i]
            target_list = chosen_target_list[i][:k_show + 1]
            pred_list = chosen_pred_list[i][:k_show + 1]
            k_dists = chosen_k_distance[i][:k_show + 1]
            self.visualize_result(chosen_post, post_image_dict, post_title_dict, target_list, pred_list, k_dists, image_source, result_dir, k_show)

    def visual_pred(self, query_index_id_list, csv_file, threshold, device, image_source, result_dir, num_image=50, k_show=6,
                    dataloader=None, embedding_path=None, show_title=True, show_image=False):

        if os.path.isfile(embedding_path):
            with open(embedding_path, 'rb') as f:
                embedding_list = pickle.load(f)
        else:
            embedding_list = self.get_all_embeddings(dataloader, device)

        posting_id_list, image_list, title_list, target_list = self.process_csv_file_for_visualization(csv_file, test_mode=False)
        post_image_dict = dict(zip(posting_id_list, image_list))
        k_post_neighbor_pred_list, k_dist_list = self.get_k_neighbors(embedding_list, posting_id_list, threshold=threshold, return_dist=True)
        post_title_dict = dict(zip(posting_id_list, title_list))

        chosen_posting_id_list = posting_id_list[query_index_id_list]
        chosen_target_list = target_list[query_index_id_list]
        chosen_pred_list = k_post_neighbor_pred_list[query_index_id_list]
        chosen_k_distance = k_dist_list[query_index_id_list]

        if show_title and show_image:
            for i in range(num_image):
                chosen_post = chosen_posting_id_list[i]
                target_list = chosen_target_list[i][:k_show + 1]
                pred_list = chosen_pred_list[i][:k_show + 1]
                k_dists = chosen_k_distance[i][:k_show + 1]
                self.visualize_result(chosen_post, post_image_dict, post_title_dict, target_list, pred_list, k_dists, image_source,
                                      result_dir, k_show)
        else:
            total_str = []
            for i in range(num_image):
                chosen_post = chosen_posting_id_list[i]
                total_str.append(str(chosen_post))
                total_str.append(post_title_dict[chosen_post])
                total_str.append("*"*10)
                target_list = chosen_target_list[i][:k_show + 1]
                target_title = [post_title_dict[target_id] for target_id in target_list]
                target_post_id_title = list(zip(target_list, target_title))
                target_title_str = [f'{str(post_id)} --- {title}'for post_id, title in target_post_id_title]
                target_title_str = '\n'.join(target_title_str)
                total_str.append(target_title_str)
                total_str.append('*'*10)
                pred_list = chosen_pred_list[i][:k_show + 1]
                pred_title = [post_title_dict[pred_id] for pred_id in pred_list]
                k_dists = chosen_k_distance[i][:k_show + 1]
                pred_id_title_dist = list(zip(pred_list, k_dists, pred_title))
                pred_id_title_dist = [f'{str(id)} --- {str(dist)} --- {title}'for id, dist, title in pred_id_title_dist]
                pred_id_title_dist = '\n'.join(pred_id_title_dist)
                total_str.append(pred_id_title_dist)
                total_str.append('*' * 20)
            total_str = '\n'.join(total_str)
            print(total_str)
            with open(os.path.join(result_dir, 'title_pred.txt'), 'w') as f:
                f.write(total_str)

    def visualize_result(self, chosen_post_id, post_image_dict, post_title_dict, target_list, pred_list, k_dist_list, image_source, result_dir, k_show):
        file_path = os.path.join(result_dir, f"{chosen_post_id}.jpg")
        query_image = post_image_dict[chosen_post_id]
        query_image = plt.imread(os.path.join(image_source, query_image))
        fig, ax = plt.subplots(nrows=3, ncols=k_show + 1, figsize=(20, 5))
        np.vectorize(lambda ax: ax.axis('off'))(ax)
        ax[0][0].imshow(query_image)
        for idx, target in enumerate(target_list):
            image_path = post_image_dict[target]
            gt_image = plt.imread(os.path.join(image_source, image_path))
            ax[1][idx].imshow(gt_image)

        for idx, pred in enumerate(pred_list):
            image_path = post_image_dict[pred]
            title = post_title_dict[pred]
            dist = k_dist_list[idx]
            pred_image = plt.imread(os.path.join(image_source, image_path))
            ax[2][idx].imshow(pred_image)
            title_with_return = ""
            for i, ch in enumerate(title):
                title_with_return += ch
                if (i != 0) & (i % 20 == 0): title_with_return += '\n'
            ax[2][idx].set_title(f"{str(dist)}\n {title_with_return}", fontsize=7)
        plt.tight_layout()
        plt.savefig(file_path)
        plt.close()

