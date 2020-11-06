"""
This file has the purpose of taking a photo, resizing it, and then creating a vector
representing the color of each pixel
Author: Alex Sutay
"""


import numpy as np
import scipy.io as sio
from PIL import Image
import os
SIZE = 40  # How many pixels should it be?


def main():
    directory = input("Where are the photos?")
    colors = {to_nums_r: "red", to_nums_g: "green", to_nums_b: "blue", to_nums_l: "grayscale"}
    for function in colors:
        outmatrix = None
        outdict = dict()
        print("Collecting data for " + colors[function])
        for filename in os.listdir(directory):
            if filename.endswith((".jpg", ".png", ".jpeg")):
                im = Image.open(directory + "/" + filename, 'r')
                this_array = function(im, SIZE, SIZE)
                im.close()
                if outmatrix is None:
                    outmatrix = this_array
                elif np.size(outmatrix) == SIZE**2:
                    outmatrix = np.asarray([outmatrix, this_array])
                else:
                    outmatrix = np.vstack([outmatrix, this_array])
        filename = directory + "/" + input("Where should I save the output?") + ".mat"
        outdict['X'] = outmatrix.astype(float)
        sio.savemat(filename, outdict)


def to_nums_r(image, height, width):
    image = image.resize((height, width))
    rgb_im = image.convert('RGB')
    r = (rgb_im.getdata(0))
    return np.asarray(list(r))


def to_nums_g(image, height, width):
    image = image.resize((height, width))
    rgb_im = image.convert('RGB')
    g = rgb_im.getdata(1)
    return np.asarray(list(g))


def to_nums_b(image, height, width):
    image = image.resize((height, width))
    rgb_im = image.convert('RGB')
    b = rgb_im.getdata(2)
    return np.asarray(list(b))


def to_nums_l(image, height, width):
    image = image.resize((height, width))
    l_im = image.convert('L')
    l = l_im.getdata()  # L is the black and white version
    return np.asarray(list(l))


if __name__ == "__main__":
    main()
