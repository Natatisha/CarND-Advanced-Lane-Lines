import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math


def draw_lines():
    return None


def show_img_grid(folder_path, num_columns=3, figsize=(16, 16)):
    images = os.listdir(folder_path)
    fig = plt.figure(figsize=figsize)
    rows = math.ceil(len(images) / num_columns)
    for i, img_path in enumerate(images):
        path = folder_path + "/" + img_path
        img = mpimg.imread(path)
        fig.add_subplot(rows, num_columns, i + 1)
        plt.imshow(img)


def compare_two_img(img1, img2):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    ax1.imshow(img1)
    ax1.set_title('Before', fontsize=30)
    ax2.imshow(img2)
    ax2.set_title('After', fontsize=30)
