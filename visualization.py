import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.cm as cm
import math
import numpy as np
import cv2
from enum import Enum
import glob


class Direction(Enum):
    LEFT = 0
    RIGHT = 1


def draw_lines():
    return None


def show_img_grid(folder_path, num_columns=3, figsize=(16, 16)):
    images = [f for f in glob.glob(folder_path + "/*.jpg")]
    fig = plt.figure(figsize=figsize)
    rows = math.ceil(len(images) / num_columns)
    for i, img_path in enumerate(images):
        img = mpimg.imread(img_path)
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
    plt.plot(left_fitx, ploty, color='r')
    plt.plot(right_fitx, ploty, color='r')


def draw_lane(image, warped_image, Minv, left_fit, right_fit):
    ploty = np.linspace(0, warped_image.shape[0] - 1, warped_image.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    warp_zero = np.zeros_like(warped_image).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
    cv2.polylines(color_warp, np.int_([pts_left]), False, (255, 0, 0), 6)
    cv2.polylines(color_warp, np.int_([pts_right]), False, (0, 0, 255), 6)

    newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0]))
    result = cv2.addWeighted(image, 1, newwarp, 0.3, 0)
    return result


def display_pos_and_curvature(image, shift_meters: float, direction: Direction, curv_rad: float):
    curv_text = "Radius of lane curvature: {:.2f}m".format(curv_rad)
    curv_text_org = (int(image.shape[1] * .05), int(image.shape[0] * .1))
    pos_text = "Vehicle is {:.2f}m {} from center".format(shift_meters, direction.name.lower())
    pos_text_org = (int(image.shape[1] * .05), int(image.shape[0] * .2))
    cv2.putText(image, curv_text, curv_text_org, cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(image, pos_text, pos_text_org, cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3, cv2.LINE_AA)
    return image


def display_text(image, text):
    text_org = (int(image.shape[1] * .05), int(image.shape[0] * .1))
    cv2.putText(image, text, text_org, cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3, cv2.LINE_AA)
    return image
