from bert.tokenization_bert import BertTokenizer
import json
from collections import defaultdict
from pathlib import Path
from typing import Union, Tuple, List

import imageio
import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader

from args import get_parser
from data.dataset_refer_bert import ReferDataset
from data.refer_data_module import get_transform, get_inv_transform, get_dataset, ReferDataModule, fetch_data_loaders
from lavt import LAVT


def mkdir(p: Union[str, Path]) -> Path:
    if isinstance(p, str):
        p = Path(p)
    if not p.exists():
        p.mkdir(parents=True)
    return p


class Cache:
    def __init__(self, num_cache: int = 100):
        parser = get_parser()
        self.args = parser.parse_args()
        self.output_dir = mkdir('./.cache')
        self.num_cache = num_cache
        if self.exists():
            self.dataset = None
        else:
            self.dataset = ReferDataset(args=self.args, image_transforms=get_transform(self.args))

    @property
    def filename_fmt(self) -> str:
        return '{:03d}.pt'

    def filepath(self, index: int) -> Path:
        return self.output_dir / self.filename_fmt.format(index)

    def exists(self) -> bool:
        if not self.output_dir.exists():
            return False
        for i in range(self.num_cache):
            if not self.filepath(i).exists():
                return False
        return True

    def fetch_cache(self) -> List[Tuple[Tensor, Tensor, Tensor, Tensor]]:
        if self.exists():
            return [torch.load(str(self.filepath(i))) for i in range(self.num_cache)]
        else:
            items = []
            for i in range(self.num_cache):
                item = self.dataset[i]
                torch.save(item, str(self.filepath(i)))
                items.append(item)
            return items


def decode_embeddings():
    from bert.tokenization_bert import BertTokenizer
    args = get_parser().parse_args()
    tokenizer = BertTokenizer.from_pretrained(args.bert_tokenizer)
    print(tokenizer.decode([101, 2157, 2611, 2006, 2723, 102], skip_special_tokens=True))


def visualize_cached_items():
    inv_transforms = get_inv_transform()
    items = Cache().fetch_cache()
    for index, (image, target, embeddings, mask) in enumerate(items[:10]):
        image, _ = inv_transforms(image, target)
        image = image.permute(1, 2, 0)
        image = (image * 255).to(dtype=torch.uint8).numpy()
        binary = (target == 1).numpy().astype(np.uint8) * 255
        iim = Image.fromarray(image)
        bim = Image.fromarray(binary)
        iim.save('./.cache/{:03d}_image.png'.format(index))
        bim.save('./.cache/{:03d}_mask.png'.format(index))


def inspect_cached_items():
    items = Cache().fetch_cache()
    for index, (image, target, embeddings, mask) in enumerate(items[:10]):
        print(image.shape, image.dtype)
        print(target.shape, target.dtype)
        print(embeddings.shape, embeddings.dtype)
        print(mask.shape, mask.dtype)

        if index > 4:
            break


def test_items_from_dataset():
    parser = get_parser()
    args = parser.parse_args()
    dataset, _ = get_dataset(split='train', args=args, eval_mode=True)
    img, target, tensor_embeddings, attention_mask = dataset[1]
    print(img.shape, img.dtype)
    print(target.shape, target.dtype)
    print(tensor_embeddings.shape, tensor_embeddings.dtype)
    print(attention_mask.shape, attention_mask.dtype)

    # print(tensor_embeddings)
    # print(attention_mask)
    #
    # from bert.tokenization_bert import BertTokenizer
    # tokenizer = BertTokenizer.from_pretrained(args.bert_tokenizer)
    # print(tokenizer.decode([101, 1996, 3203, 2007, 1996, 2630, 3797, 102], skip_special_tokens=True))
    # print(tokenizer.decode([101, 3203, 1059, 2067, 2000, 2149, 102], skip_special_tokens=True))
    # print(tokenizer.decode([101, 2630, 3797, 102], skip_special_tokens=True))


def run_custom_test_from_checkpoint():
    parser = get_parser()
    args = parser.parse_args()
    dataset, _ = get_dataset(split='test', args=args, eval_mode=True)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    device = 'cuda:0'

    assert args.resume
    print('Load the model from {}'.format(args.resume))
    model = LAVT.load_from_checkpoint(args.resume, args=args, num_train_steps=0)
    model.cuda(device=device)

    inv_transforms = get_inv_transform()
    from bert.tokenization_bert import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained(args.bert_tokenizer)

    with torch.no_grad():
        for i, data in enumerate(dataloader):
            image, target, sentences, attentions = data
            image, target, sentences, attentions = image.to(device), target.to(device), \
                                                   sentences.to(device), attentions.to(device)
            print(image.shape, target.shape, sentences.shape, attentions.shape)
            print(image.dtype, target.dtype, sentences.dtype, attentions.dtype)
            # (bsize, 3, img_size, img_size)
            # (bsize, img_size, img_size)
            # (bsize, 1, max_len, num_anno)
            # (bsize, 1, max_len, num_anno)

            sentences = sentences.squeeze(1)
            attentions = attentions.squeeze(1)
            target = target.cpu().data.numpy()

            image_, _ = inv_transforms(image, target)
            image_ = image_[0, ...].permute(1, 2, 0).cpu()
            image_ = (image_ * 255).to(dtype=torch.uint8).numpy()
            imageio.imwrite('visualization/{:05d}_image.png'.format(i), image_)
            sentence_list = []

            for j in range(sentences.size(-1)):
                last_hidden_states = model.bert_model(sentences[:, :, j], attention_mask=attentions[:, :, j])[0]
                embedding = last_hidden_states.permute(0, 2, 1)
                output = model.model(image, embedding, l_mask=attentions[:, :, j].unsqueeze(-1))
                output = output.cpu()
                output_mask = output.argmax(1).data.numpy()

                # print(output_mask.shape, output_mask.dtype, np.min(output_mask), np.max(output_mask))
                # print(target.shape, target.dtype, np.min(target), np.max(target))
                sentence_ = sentences[:, :, j].cpu().data.numpy().tolist()[0]
                sentence = tokenizer.decode(sentence_, skip_special_tokens=True)
                sentence_list.append(sentence)
                imageio.imwrite('visualization/{:05d}_pred{}.png'.format(i, j), output_mask[0, ...].astype(np.uint8) * 255)
                imageio.imwrite('visualization/{:05d}_targ.png'.format(i), target[0, ...].astype(np.uint8) * 255)
            with open('visualization/{:05d}_sen.txt'.format(i), 'w') as file:
                file.write('\n'.join(sentence_list))

            if i > 9:
                break

def convert_sunrefer_to_lavt():
    import cv2

    args = get_parser().parse_args()
    transforms = get_transform(args)
    inv_transforms = get_inv_transform()
    max_len = 50

    device = 'cuda:0'
    in_refer_path = Path('/home/junha/projects/Refer-it-in-RGBD/data/sunrefer_singleRGBD/SUNREFER_v2.json')
    in_image_dir = Path('/home/junha/projects/Refer-it-in-RGBD/sunrgbd/tmp_seg/high_confidence')

    # Read refer data.
    with open(str(in_refer_path), 'r') as file:
        refer_data = json.load(file)

    refer_by_image_id = defaultdict(list)
    for refer_item in refer_data:
        refer_by_image_id[refer_item['image_id']].append(refer_item['sentence'])

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    print('Load the model from {}'.format(args.resume))
    model = LAVT.load_from_checkpoint(args.resume, args=args, num_train_steps=0)
    model.cuda(device=device)

    mask_path_list = sorted(in_image_dir.glob('*.png'))
    for mask_path in mask_path_list:
        image_id = mask_path.stem
        anno = refer_by_image_id[image_id]
        if not anno:
            continue

        image_path = in_image_dir / '{}.jpg'.format(image_id)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        mask = Image.fromarray(mask.astype(np.uint8), mode='P')
        image = Image.open(str(image_path)).convert('RGB')

        # resize, from PIL to tensor, and mean and std normalization
        image, target = transforms(image, mask)

        encoded_anno = tokenizer.batch_encode_plus(anno, max_length=max_len, truncation=True, pad_to_max_length=True)
        sentences = torch.tensor(encoded_anno['input_ids']).permute(1, 0)
        attentions = torch.tensor(encoded_anno['attention_mask']).permute(1, 0)

        image = image.unsqueeze(0)
        target = target.unsqueeze(0)
        sentences = sentences.unsqueeze(0)
        attentions = attentions.unsqueeze(0)

        image, target, sentences, attentions = image.to(device), target.to(device), \
                                               sentences.to(device), attentions.to(device)
        print(image.shape, target.shape, sentences.shape, attentions.shape)
        print(image.dtype, target.dtype, sentences.dtype, attentions.dtype)
        # (bsize, 3, img_size, img_size)
        # (bsize, img_size, img_size)
        # (bsize, 1, max_len, num_anno)
        # (bsize, 1, max_len, num_anno)

        target = target.cpu().data.numpy()

        image_, _ = inv_transforms(image, target)
        image_ = image_[0, ...].permute(1, 2, 0).cpu()
        image_ = (image_ * 255).to(dtype=torch.uint8).numpy()
        imageio.imwrite('visualization/{}_image.png'.format(image_id), image_)
        sentence_list = []

        for j in range(sentences.size(-1)):
            last_hidden_states = model.bert_model(sentences[:, :, j], attention_mask=attentions[:, :, j])[0]
            embedding = last_hidden_states.permute(0, 2, 1)
            output = model.model(image, embedding, l_mask=attentions[:, :, j].unsqueeze(-1))
            output = output.cpu()
            output_mask = output.argmax(1).data.numpy()

            # print(output_mask.shape, output_mask.dtype, np.min(output_mask), np.max(output_mask))
            # print(target.shape, target.dtype, np.min(target), np.max(target))
            sentence_ = sentences[:, :, j].cpu().data.numpy().tolist()[0]
            sentence = tokenizer.decode(sentence_, skip_special_tokens=True)
            sentence_list.append(sentence)
            imageio.imwrite('visualization/{}_pred{}.png'.format(image_id, j), output_mask[0, ...].astype(np.uint8) * 255)
            imageio.imwrite('visualization/{}_targ.png'.format(image_id), target[0, ...].astype(np.uint8) * 255)
        with open('visualization/{}_sen.txt'.format(image_id), 'w') as file:
            file.write('\n'.join(sentence_list))


def check_dataset():
    args = get_parser().parse_args()
    train_dl, val_dl, test_dl = fetch_data_loaders(args)
    model = LAVT(args=args, num_train_steps=len(train_dl))

    import pytorch_lightning as pl
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        gpus=args.gpus,
        strategy='ddp',
        sync_batchnorm=True,
        num_sanity_val_steps=0)
    trainer.fit(model=model, train_dataloaders=train_dl, val_dataloaders=val_dl)


if __name__ == '__main__':
    check_dataset()
    # convert_sunrefer_to_lavt()
    # run_custom_test_from_checkpoint()
