from pathlib import Path
from typing import Union, Tuple, List

import numpy as np
import torch
from PIL import Image
from torch import Tensor

from args import get_parser
from data.dataset_refer_bert import ReferDataset
from data.refer_data_module import get_transform, get_inv_transform


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


def main():
    items = Cache().fetch_cache()
    for index, (image, target, embeddings, mask) in enumerate(items[:10]):
        print(image.shape, image.dtype)
        print(target.shape, target.dtype)
        print(embeddings.shape, embeddings.dtype)
        print(mask.shape, mask.dtype)

        if index > 4:
            break


if __name__ == '__main__':
    main()
