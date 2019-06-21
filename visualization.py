import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.cm as cm
import math
import numpy as np


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


def compare_with_binary(img, binary_img):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    ax1.imshow(img)
    ax1.set_title('Before', fontsize=30)
    ax2.imshow(binary_img, cmap=cm.gray)
    ax2.set_title('After', fontsize=30)


def draw_poly_on_img(img, points):
    plt.imshow(img)
    polygon = plt.Polygon(points, fill=None, edgecolor='r')
    plt.gca().add_patch(polygon)


def display_polynomial(img, left_fit, right_fit):
    ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])
    try:
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1 * ploty ** 2 + 1 * ploty
        right_fitx = 1 * ploty ** 2 + 1 * ploty

    # Plots the left and right polynomials on the lane lines
    plt.imshow(img, cmap=cm.gray)
    plt.plot(left_fitx, ploty, color='red')
    plt.plot(right_fitx, ploty, color='red')
