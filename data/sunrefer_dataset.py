import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import BertTokenizer

from data.instance_storage import InstanceStorage
from utils import transforms as T


def create_train_val_id_list():
    root_dir = Path('/home/junha/projects/Refer-it-in-RGBD/sunrgbd/tmp_seg')
    high_conf_dir = root_dir / 'high_confidence'
    seg_path_list = sorted(high_conf_dir.glob('*.png'))

    image_id_list = list(map(lambda x: x.stem, seg_path_list))
    train_id_list, val_id_list = [], []
    for image_id in image_id_list:
        idx = int(image_id)
        if idx <= 5050:
            val_id_list.append(image_id)
        else:
            train_id_list.append(image_id)
    train_out_path = root_dir / 'train_data_idx.txt'
    val_out_path = root_dir / 'val_data_idx.txt'
    with open(str(train_out_path), 'w') as file:
        file.write('\n'.join(train_id_list))
    with open(str(val_out_path), 'w') as file:
        file.write('\n'.join(val_id_list))


def get_transform(img_size: int):
    transforms = [T.Resize(img_size, img_size),
                  T.ToTensor(),
                  T.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])]
    return T.Compose(transforms)


class SUNREFERDataset(Dataset):
    def __init__(self, split: str, eval_mode: bool, image_size: int = 480, max_len: int = 50):
        Dataset.__init__(self)
        self.root_dir = Path('/home/junha/projects/Refer-it-in-RGBD/sunrgbd/tmp_seg')
        self.db_path = Path('/home/junha/projects/Refer-it-in-RGBD/sunrgbd/tmp_seg/database')
        self.refer_path = Path('/home/junha/projects/Refer-it-in-RGBD/data/sunrefer_singleRGBD/SUNREFER_v2.json')
        self.eval_mode = eval_mode

        self.db = InstanceStorage(read_only=True, db_path=self.db_path)
        self.data_idx_path = self.root_dir / '{}_data_idx.txt'.format(split)
        with open(str(self.data_idx_path), 'r') as file:
            self.data_idx_list = file.read().split()
        self.image_size = image_size
        self.transforms = get_transform(image_size)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        with open(str(self.refer_path), 'r') as file:
            refer_data = json.load(file)
        sentence_by_image_id = defaultdict(list)
        for refer_item in refer_data:
            image_id = refer_item['image_id']
            sentence = refer_item['sentence']
            sentence_by_image_id[image_id].append(sentence)

        self.refer_data = dict()
        for image_id, sentences in sentence_by_image_id.items():
            anno = self.tokenizer.batch_encode_plus(sentences, truncation=True, padding='max_length', max_length=max_len)
            sentences = torch.tensor(anno['input_ids']).permute(1, 0)
            attentions = torch.tensor(anno['attention_mask']).permute(1, 0)
            self.refer_data[image_id] = sentences, attentions

    def __len__(self):
        return len(self.data_idx_list)

    def __getitem__(self, idx):
        """
        Returns a tuple of items from the index.
        @image: float32, (3, 480, 480)
        @target: int64, (480, 480)
        @tensor_embeddings: int64, (1, 20)
        @attention_mask: int64, (1, 20)
        """
        image_id = self.data_idx_list[idx]
        rgb = Image.fromarray(self.db.get_rgb(image_id), 'RGB')
        seg = Image.fromarray(self.db.get_seg(image_id) > 0, 'P')
        image, target = self.transforms(rgb, seg)
        sentences, attentions = self.refer_data[image_id]
        if self.eval_mode:
            sentences = sentences.unsqueeze(0)
            attentions = attentions.unsqueeze(0)
        else:
            rand_idx = np.random.choice(sentences.shape[1])
            sentences = sentences[:, rand_idx]
            attentions = attentions[:, rand_idx]
        return image, target, sentences, attentions
