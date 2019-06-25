import numpy as np


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


def search_around_poly(binary_warped, left_fit, right_fit, margin=100):
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    poly_left = left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2]
    poly_right = right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2]
    left_lane_inds = (nonzerox > (poly_left - margin)) & (nonzerox < (poly_left + margin))
    right_lane_inds = (nonzerox > (poly_right - margin)) & (nonzerox < (poly_right + margin))

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


def calc_lines_dist(y_eval, left_fit, right_fit, convert_to_meters=False, lane_width_m=3.7, lane_width_pix=700):
    left_x = left_fit[0] * y_eval ** 2 + left_fit[1] * y_eval + left_fit[2]
    right_x = right_fit[0] * y_eval ** 2 + right_fit[1] * y_eval + right_fit[2]
    dist = right_x - left_x
    return dist if not convert_to_meters else dist * (lane_width_m / lane_width_pix)


def check_if_parallel(img_height, left_fit, right_fit, std_delta=70):
    points = [point for point in range(0, img_height, int(img_height / 4))]
    distances = [calc_lines_dist(p, left_fit, right_fit, convert_to_meters=False) for p in points]
    return all(i > 0 for i in distances) and np.std(distances) <= std_delta


def calc_lane_width(img_height, left_fit, right_fit, lane_width_m=3.7, lane_width_pix=700):
    points = [point for point in range(0, img_height, int(img_height / 4))]
    distances_m = [np.abs(calc_lines_dist(p, left_fit, right_fit, convert_to_meters=True, lane_width_m=lane_width_m,
                                          lane_width_pix=lane_width_pix)) for p in points]
    return np.mean(distances_m)


def calc_vehicle_shift_m(lefty, left_fit, right_fit, img_width, lane_width_m=3.7):
    y_eval = np.max(lefty)
    left_line = left_fit[0] * y_eval ** 2 + left_fit[1] * y_eval + left_fit[2]
    right_line = right_fit[0] * y_eval ** 2 + right_fit[1] * y_eval + right_fit[2]

    center_of_lane = (right_line - left_line) / 2 + left_line

    dist_left = center_of_lane
    dist_right = img_width - center_of_lane
    # scale for meters
    xm_per_pix = lane_width_m / (right_line - left_line)
    return (dist_right - dist_left) * xm_per_pix
