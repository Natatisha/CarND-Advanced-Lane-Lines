import numpy as np
import cv2


# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = []
        # average x values of the fitted line over the last n iterations
        self.bestx = None
        # polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        # polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        # radius of curvature of the line in some units
        self.radius_of_curvature = None
        # distance in meters of vehicle center from the line
        self.line_base_pos = None
        # difference in fit coefficients between last and new fits
        self.diffs = np.array([0, 0, 0], dtype='float')
        # x values for detected line pixels
        self.allx = None
        # y values for detected line pixels
        self.ally = None


def sliding_window(binary_warped_img, nwindows=9, margin=100, minpix=50):
    leftx_base, rightx_base = find_lines_basepoints(binary_warped_img)

    window_height = np.int(binary_warped_img.shape[0] // nwindows)

    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped_img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped_img.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped_img.shape[0] - window * window_height

        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin

        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        if len(good_left_inds) > minpix:
            leftx_current = np.mean(nonzerox[good_left_inds], dtype=np.int32)
        if len(good_right_inds) > minpix:
            rightx_current = np.mean(nonzerox[good_right_inds], dtype=np.int32)

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty


def fit_poly(leftx, lefty, rightx, righty):
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    return left_fit, right_fit


def find_lines_basepoints(binary_warped_img):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped_img[binary_warped_img.shape[0] // 2:, :], axis=0)
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] // 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    return leftx_base, rightx_base


def measure_curvature(leftx, lefty, righty, rightx, lane_length_m=20., lane_width_m=3.7, lane_length_pix=720,
                      lane_width_pix=700):
    ym_per_pix = lane_length_m / lane_length_pix
    xm_per_pix = lane_width_m / lane_width_pix

    left_fit = np.polyfit(lefty * ym_per_pix, leftx * xm_per_pix, 2)
    right_fit = np.polyfit(righty * ym_per_pix, rightx * xm_per_pix, 2)

    left_curverad = ((1. + (2. * left_fit[0] * np.max(lefty) * ym_per_pix + left_fit[1]) ** 2) ** 1.5) / (
        np.absolute(2. * left_fit[0]))
    right_curverad = ((1. + (2. * right_fit[0] * np.max(righty) * ym_per_pix + right_fit[1]) ** 2) ** 1.5) / (
        np.absolute(2. * right_fit[0]))

    return (left_curverad + right_curverad) / 2.


def calc_vehicle_shift_m(leftx, rightx, img_width, lane_width_m=3.7):
    # take bottom pixels that are closer to the vehicle
    left_line = np.mean(leftx[:50])
    right_line = np.mean(rightx[:50])

    center_of_lane = (right_line - left_line) / 2 + left_line

    dist_left = center_of_lane
    dist_right = img_width - center_of_lane
    # scale for meters
    xm_per_pix = lane_width_m / (right_line - left_line)
    return (dist_right - dist_left) * xm_per_pix
