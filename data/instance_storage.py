from pathlib import Path
from typing import Any, Callable, Tuple

import lmdb
import numpy as np


def encode_str(query: str) -> bytes:
    return query.encode()


def decode_str(byte_str: bytes) -> str:
    return byte_str.decode('utf-8')


def encode_array(query: np.ndarray) -> bytes:
    return query.tobytes()


def decode_array(byte_str: bytes, dtype, shape) -> np.ndarray:
    return np.frombuffer(byte_str, dtype=dtype).reshape(*shape)


def decode_point(byte_str: bytes) -> np.ndarray:
    return decode_array(byte_str, np.float32, (-1, 6))


def decode_bbox(byte_str: bytes) -> np.ndarray:
    return decode_array(byte_str, np.float32, (1, 6))


class InstanceStorage:
    def __init__(
            self,
            read_only: bool = True,
            db_path: Path = None):
        self.db_path = db_path
        self.num_dbs = 3
        self.env = lmdb.open(
            path=str(self.db_path),
            max_dbs=self.num_dbs,
            map_size=5e11,
            max_readers=1,
            readonly=read_only,
            lock=False,
            readahead=False,
            meminit=False
        )
        self.rgb = self.env.open_db(encode_str('rgb'))
        self.seg = self.env.open_db(encode_str('seg'))
        self.shape = self.env.open_db(encode_str('shape'))

    def _get_num_items(self, db) -> int:
        with self.env.begin(db=db, write=False) as txn:
            return txn.stat()['entries']

    def __len__(self):
        return self._get_num_items(self.rgb)

    def _get_item(
            self,
            db,
            image_id: str,
            decode_func: Callable[[Any], Any]):
        with self.env.begin(db=db, write=False) as txn:
            return decode_func(txn.get(encode_str(image_id)))

    def _put_item(
            self,
            db,
            image_id: str,
            item,
            encode_func):
        with self.env.begin(db=db, write=True) as txn:
            txn.put(encode_str(image_id), encode_func(item))

    def put_shape(self, image_id: str, shape: Tuple[int, int]):
        self._put_item(
            db=self.shape,
            image_id=image_id,
            item='{},{}'.format(*shape),
            encode_func=encode_str)

    def get_shape(self, image_id: str) -> Tuple[int, int]:
        res = self._get_item(
            db=self.shape,
            image_id=image_id,
            decode_func=decode_str)
        s1, s2 = res.split(',')
        return int(s1), int(s2)

    def get_rgb(self, image_id: str) -> np.ndarray:
        """
        Returns an RGB image.
        :param image_id: string key.
        :return: an RGB image in np.ndarray (H, W, 3).
        """
        s1, s2 = self.get_shape(image_id)
        return self._get_item(
            db=self.rgb,
            image_id=image_id,
            decode_func=lambda x: decode_array(x, dtype=np.uint8, shape=(s1, s2, 3)))

    def put_rgb(self, image_id: str, rgb: np.ndarray):
        self._put_item(
            db=self.rgb,
            image_id=image_id,
            item=rgb,
            encode_func=encode_array)
        self.put_shape(image_id=image_id, shape=(rgb.shape[0], rgb.shape[1]))

    def get_seg(self, image_id: str) -> np.ndarray:
        s1, s2 = self.get_shape(image_id)
        return self._get_item(
            db=self.seg,
            image_id=image_id,
            decode_func=lambda x: decode_array(x, dtype=np.uint8, shape=(s1, s2)))

    def put_seg(self, image_id: str, seg: np.ndarray):
        self._put_item(
            db=self.seg,
            image_id=image_id,
            item=seg,
            encode_func=encode_array)

    def __del__(self):
        self.env.close()
