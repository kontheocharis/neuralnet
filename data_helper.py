from glob import glob
import tensorflow as tf
from typing import List
import os
import random


class DataHelper:
    _dirname = os.path.dirname(__file__)
    _training_filenames: List[str] = []

    def __init__(self, train_size: int = 1000,
                 test_size: int = 4000,
                 classes: List[str] = []):
        total = self.get_total_examples()
        assert train_size + test_size <= total
        self.train_size: int = train_size
        self.test_size: int = test_size

        all_classes = os.listdir(self.get_path('data/'))
        assert set(classes).issubset(all_classes)
        self.classes: List[str] = all_classes \
            if len(classes) == 0 else classes
        self.train_filenames, self.test_filenames = self._split_train_test()

    def get_dataset(self, training: bool = False) -> tf.data.Dataset:
        l_filenames, (a_filenames, b_filenames) = self._get_filenames(training)
        l_images, (a_images, b_images) = self._parse_images(
                l_filenames, a_filenames, b_filenames)
        return tf.data.Dataset.from_tensor_slices((
                l_images, a_images, b_images))

    def _split_train_test(self) -> ([str], [str]):
        classes = '{' + ','.join(self.classes) + '}' \
                if len(self.classes) > 1 else self.classes[0]
        all_filenames = glob(self.get_path(f'data/{classes}/*.jpg'))
        all_filenames = ['/'.join(os.path.abspath(i).split('/')[-2:])
                         for i in all_filenames]
        random.shuffle(all_filenames)
        return all_filenames[:self.train_size], all_filenames[-self.test_size:]

    def _parse_images(self, l_filenames, a_filenames, b_filenames
                      ) -> (tf.Tensor, (tf.Tensor, tf.Tensor)):
        l_decoded = [self._decode(x) for x in l_filenames]
        a_decoded = [self._decode(x) for x in a_filenames]
        b_decoded = [self._decode(x) for x in b_filenames]
        return l_decoded, (a_decoded, b_decoded)

    def _decode(self, x):
        tf.image.decode_jpeg(tf.read_file(x))

    def _get_filenames(self, training: bool = False
                       ) -> (str, (str, str)):
        filenames = self.train_filenames if training else self.test_filenames
        l_filenames = [
            self.get_path(f'transformed_data/l/{name}')
            for name in filenames
        ]

        a_filenames = [
            self.get_path(f'transformed_data/a/{name}')
            for name in filenames
        ]

        b_filenames = [
            self.get_path(f'transformed_data/b/{name}')
            for name in filenames
        ]
        return l_filenames, (a_filenames, b_filenames)

    def get_total_examples(self) -> int:
        return len(glob(self.get_path('data/*/*.jpg')))

    def get_path(self, path: str) -> str:
        return os.path.join(self._dirname, path)

