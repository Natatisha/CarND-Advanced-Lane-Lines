import cv2
import numpy as np
import pickle


def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if orient == 'x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    elif orient == 'y':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    else:
        raise Exception("Orient param should be either x or y!")

    abs_sobel = np.absolute(sobel)
    scaled = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    sobel_binary = np.zeros(scaled.shape)
    sobel_binary[(scaled >= thresh[0]) & (scaled <= thresh[1])] = 1
    return sobel_binary


def mag_thresh(img, sobel_kernel=3, thresh=(0, 255)):
    sobel_x, sobel_y = sobel_xy(img, sobel_kernel)
    abs_sobel = np.sqrt(np.square(sobel_x) + np.square(sobel_y))
    scaled = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    mag_binary = np.zeros(scaled.shape)
    mag_binary[(scaled >= thresh[0]) & (scaled <= thresh[1])] = 1
    return mag_binary


def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi / 2)):
    sobel_x, sobel_y = sobel_xy(img, sobel_kernel)

    abs_sobel_x = np.absolute(sobel_x)
    abs_sobel_y = np.absolute(sobel_y)
    scaled = np.arctan2(abs_sobel_y, abs_sobel_x)
    dir_binary = np.zeros(scaled.shape)
    dir_binary[(scaled >= thresh[0]) & (scaled <= thresh[1])] = 1
    return dir_binary


def sobel_xy(img, sobel_kernel):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    return sobel_x, sobel_y


def sobel_mag_dir_combined_threshold(img, sobelx_thresh=(20, 110), sobely_thresh=(50, 200), magnitude_thresh=(40, 150),
                                     sobel_kernel=3, dir_thresh=(0.7, 1.3), dir_sobel_kernel=15):
    gradx = abs_sobel_thresh(img, orient='x', sobel_kernel=sobel_kernel, thresh=sobelx_thresh)
    grady = abs_sobel_thresh(img, orient='y', sobel_kernel=sobel_kernel, thresh=sobely_thresh)
    mag_binary = mag_thresh(img, sobel_kernel=sobel_kernel, thresh=magnitude_thresh)
    dir_binary = dir_threshold(img, sobel_kernel=dir_sobel_kernel, thresh=dir_thresh)

    combined = np.zeros_like(dir_binary)
    combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
    return combined


def gradient_threshold(img):
    return abs_sobel_thresh(img, 'x', 3, (20, 110))


def s_channel_threshold(img, thresh=(170, 255)):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:, :, 2]
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= thresh[0]) & (s_channel <= thresh[1])] = 1
    return s_binary


def threshold(img):
    grad = gradient_threshold(img)
    s_channel = s_channel_threshold(img)
    combined_binary = np.zeros_like(grad)
    combined_binary[(s_channel == 1) | (grad == 1)] = 1
    return combined_binary


def save_perspective_points(src_points, dst_points, file_path='saved_checkpoints/perspective'):
    dist_pickle = {"src": src_points, "dst": dst_points}
    pickle.dump(dist_pickle, open(file_path + ".p", "wb"))


def load_perspective_points(file_path='saved_checkpoints/perspective.p'):
    dist_pickle = pickle.load(open(file_path, "rb"))
    src = dist_pickle["src"]
    dst = dist_pickle["dst"]
    return src, dst


def perspective_transform(img, src_points, dst_points):
    img_size = (img.shape[1], img.shape[0])
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    return cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
