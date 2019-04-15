import tensorflow as tf
import os
import random
import itertools
from glob import glob
import numpy as np

_dirname = os.path.dirname(__file__)


class DataHelper:
    categories = os.listdir('data')

    def __init__(self, size=-1, image_dim=256, category_glob='*'):
        self.image_dim = image_dim
        # all_filenames = glob('data/**/*.jpg')
        self.size = size
        self.category_glob = category_glob
        # self.filenames = np.array(random.sample(all_filenames, self.size))

    def get_dataset(self):
        # filename_categories = np.array([x.split('/')[1] for x in self.filenames])
        # filename_category_indices = np.array([tf.one_hot(self.categories.index(x), len(self.categories)) for x in filename_categories])

        # return tf.data.Dataset.from_tensor_slices(
        #         (self.filenames, filename_categories, filename_category_indices)
        #         ) \
        #         .map(self._apply_to_each_elem) \

        return tf.data.Dataset.list_files('data/' + self.category_glob + '/*.jpg').take(self.size).map(self._apply_to_each_elem)

    def _apply_to_each_elem(self, filename):
        return (self._decode(filename), self._get_category(filename),filename)

    def _get_category(self, filename):
        onehot = tf.one_hot(*tf.contrib.eager.py_func(
                func=lambda x: (self.categories.index(str(x).split('/')[1]),
                    len(self.categories)),
                inp=[filename],
                Tout=(tf.int32, tf.int32)
                ))
        onehot.set_shape((len(self.categories),))
        return onehot

    def _decode(self, x):
        img = tf.image.decode_jpeg(tf.read_file(x))
        img.set_shape([self.image_dim, self.image_dim, 3])
        into_float = tf.to_float(img)
        return tf.math.divide(
                into_float,
                tf.constant(255.0, dtype='float32'))

def test():
    h = DataHelper()
    dataset = h.get_dataset()
    it = dataset.make_one_shot_iterator()
    with tf.Session() as sess:
        print(sess.run(it.get_next()))

# test()

# class OLDDataHelper:
#     _dirname = os.path.dirname(__file__)
#     _training_filenames: List[str] = []
#     image_dim = 256

#     def __init__(self, train_size: int = 1000,
#                  test_size: int = 4000,
#                  classes: List[str] = []):
#         total = self.get_total_examples()
#         assert train_size + test_size <= total
#         self.train_size: int = train_size
#         self.test_size: int = test_size

#         all_classes = os.listdir(self.get_path('data/'))
#         assert set(classes).issubset(all_classes)
#         self.classes: List[str] = all_classes \
#             if len(classes) == 0 else classes
#         self.train_filenames, self.test_filenames = self._split_train_test()

#     def get_dataset(self, training: bool = False) -> tf.data.Dataset:
#         l_filenames, (a_filenames, b_filenames) = self._get_filenames(training)
#         l_images, (a_images, b_images) = self._parse_images(
#                 l_filenames, a_filenames, b_filenames)
#         return tf.data.Dataset.from_tensor_slices((
#                 l_images, [a_images, b_images]))

#     def _split_train_test(self) -> ([str], [str]):
#         classes = '{' + ','.join(self.classes) + '}' \
#                 if len(self.classes) > 1 else self.classes[0]
#         all_filenames = glob(self.get_path(f'data/{classes}/*.jpg'))
#         all_filenames = ['/'.join(os.path.abspath(i).split('/')[-2:])
#                          for i in all_filenames]
#         random.shuffle(all_filenames)
#         return all_filenames[:self.train_size], all_filenames[-self.test_size:]

#     def _parse_images(self, l_filenames, a_filenames, b_filenames
#                       ) -> (tf.Tensor, (tf.Tensor, tf.Tensor)):
#         l_decoded, a_decoded, b_decoded = (
#             [self._decode(x) for x in filenames]
#             for filenames in [l_filenames, a_filenames, b_filenames]
#         )
#         return l_decoded, (a_decoded, b_decoded)

#     def _decode(self, x) -> tf.Tensor:
#         return tf.image.decode_jpeg(tf.read_file(x))

#     def _get_filenames(self, training: bool = False
#                        ) -> (str, (str, str)):
#         filenames = self.train_filenames if training else self.test_filenames
#         l_filenames, a_filenames, b_filenames = (
#             [self.get_path(f'transformed_data/{x}/{name}')
#                 for name in filenames]
#             for x in 'lab'
#         )
#         return l_filenames, (a_filenames, b_filenames)

#     def get_total_examples(self) -> int:
#         return len(glob(self.get_path('data/*/*.jpg')))

#     def get_path(self, path: str) -> str:
#         return os.path.join(self._dirname, path)
