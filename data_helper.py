import cProfile
import pstats
from io import StringIO
import tensorflow as tf
import os
import math
import random

_dirname = os.path.dirname(__file__)


class DataHelper:
    image_dim = 256
    categories = os.listdir(os.path.join(_dirname, 'data'))

    def __init__(self, train_size=1000, test_size=4000, categories=None):
        if categories:
            self.categories = categories
        self._training_filenames = {c: [] for c in self.categories}
        self._train_size = train_size
        self._test_size = test_size
        self._train_category_sizes = self._get_size_per_category(train_size)
        self._test_category_sizes = self._get_size_per_category(test_size)

    def get_dataset(self, training=False):
        dataset_list = [[], []]
        for (index_c, c) in enumerate(self.categories):
            size = self._train_category_sizes[index_c] if \
                training else self._test_category_sizes[index_c]
            category_dir = os.path.join(_dirname, 'data', c)
            image_filenames = []
            if training:
                image_filenames = random.sample(
                        os.listdir(category_dir), size)
            else:
                image_filenames = random.sample(
                        [x for x in os.listdir(category_dir)
                            if x not in self._training_filenames[c]],
                        size)
            if training:
                self._training_filenames[c] = []
            for i in image_filenames:
                image = os.path.join(_dirname, 'data', c, i)

                dataset_list[0].append(
                        self._create_category_list(
                            len(self.categories), index_c))

                dataset_list[1].append(self._decode(image))
                if training:
                    self._training_filenames[c].append(image)
        print('done')
        return tf.data.Dataset.from_tensor_slices((
            dataset_list[1], dataset_list[0]))

    def _decode(self, x):
        img = tf.image.decode_jpeg(tf.read_file(x))
        img.set_shape([self.image_dim, self.image_dim, 3])
        reshaped = tf.reshape(tf.to_float(img), [-1])
        return tf.math.divide(
                reshaped,
                tf.constant(255.0, dtype='float32'))

    def _create_category_list(self, length, category):
        return [1.0 if i == category else 0 for i in range(length)]

    def _get_size_per_category(self, size):
        num_per_category = math.ceil(size / len(self.categories))
        sizes = []
        while True:
            sm = sum(sizes)
            if sm == size:
                break
            elif sm + num_per_category > size:
                sizes.append(size - sm)
                break
            else:
                sizes.append(num_per_category)
        return sizes


def test():
    pr = cProfile.Profile()
    pr.enable()  # start profiling

    h = DataHelper()
    dataset = h.get_dataset(training=True)
    it = dataset.make_one_shot_iterator()
    with tf.Session() as sess:
        items = []
        for i in range(5):
            items.append(it.get_next())
        print(sess.run(items))

    pr.disable()  # end profiling
    s = StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print(s.getvalue())

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
