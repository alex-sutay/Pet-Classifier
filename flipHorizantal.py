"""
This file is designed to artificially inflate a data set by mirroring all of the images horizontally
Author: Alex Sutay
"""

from PIL import Image, ImageOps
import os


def main():
    """
    Prompt the user for a directory and then flip all of the images horizontally
    :return: None
    """
    directory = input("Where are the photos?")
    print("Flipping all images...")
    counter = 0
    for filename in os.listdir(directory):
        if filename.endswith((".jpg", ".png", ".jpeg")):
            counter += 1
            im = Image.open(directory + "/" + filename, 'r')
            im = ImageOps.mirror(im)
            im.save(directory + '/flip' + str(counter) + '.jpg')
            im.close()
        if counter % 100 == 0:
            print(str(counter) + ' done...')
    print("All done!")


if __name__ == "__main__":
    main()
