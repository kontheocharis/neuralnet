"""Convert Images

This script converts all RGB images in ./data/*/*.jpg into LAB
and puts them into ./transformed_data/{a,b,c}/*.jpg.
"""

from skimage import io, color
import cv2
# import sys
import os
import glob
import warnings
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
# import numpy as np


number_of_images = 5000


def main():
    dirname = os.path.dirname(__file__)
    filenames = glob.glob(
        os.path.join(dirname, 'data/*/*.jpg'))[:number_of_images]

    for filename in filenames:
        file = read_file(filename)
        l, a, b = transform_image(file)
        lp, ap, bp = resolve_lab_paths(filename, dirname)
        write_file(lp, l)
        write_file(ap, a)
        write_file(bp, b)


def transform_image(file):
    lab_file = color.rgb2lab(file)

    # Rescale LAB to [0,1]^3 (normally [0,100]x[-128,127]x[-128,127])
    lab_scaled = (lab_file + [0, 128, 128]) / [100, 255, 255]

    l, a, b = cv2.split(lab_scaled)
    return l, a, b


def read_file(filename):
    return io.imread(filename)


def write_file(filename, content):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return io.imsave(filename, content)


def resolve_lab_paths(filename, dirname):
    base_dir = filename.split(os.sep)[-2]
    filebasename = os.path.split(filename)[1]

    for i in ('l', 'a', 'b'):
        target_dir = os.path.join(dirname, 'transformed_data/'+i+'/'+base_dir)
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

    l_path = os.path.join(
        dirname, 'transformed_data/l/'+base_dir+'/'+filebasename)
    a_path = os.path.join(
        dirname, 'transformed_data/a/'+base_dir+'/'+filebasename)
    b_path = os.path.join(
        dirname, 'transformed_data/b/'+base_dir+'/'+filebasename)

    return l_path, a_path, b_path


if __name__ == '__main__':
    main()
