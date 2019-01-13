from skimage import io, color
import sys
import os

cwd = os.getcwd()
filenames = [os.path.join(cwd, i) for i in sys.argv][1:]
print(filenames)

for filename in filenames:
    file = io.imread(filename)
    lab_file = color.rgb2lab(file)
    print(lab_file)
