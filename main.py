import numpy as np
import matplotlib.image as mpimg
import camera as cam
import processing as pr
import detection as det
import visualization as vs
import cv2


def find_road_lane(img_path):
    img = mpimg.imread(img_path)
    mtx, dst = cam.load_calibration_results()
    undistort = cam.undistort(img, mtx, dst, True)

    binary = pr.threshold(undistort)

    src, dst = pr.load_perspective_points()
    M, Minv, binary_warped = pr.perspective_transform(binary, src, dst)

    leftx, lefty, rightx, righty = det.sliding_window(binary_warped)
    left_fit, right_fit = det.fit_poly(leftx, lefty, rightx, righty)

    curv_rad = det.measure_curvature(leftx, lefty, righty, rightx)
    vehicle_shift_m = det.calc_vehicle_shift_m(leftx, rightx, binary_warped.shape[1])
    res = vs.draw_lane(img, binary_warped, Minv, left_fit, right_fit)
    dir = vs.Direction.RIGHT if vehicle_shift_m >= 0 else vs.Direction.LEFT
    res_text = vs.display_pos_and_curvature(res, np.absolute(vehicle_shift_m), dir, curv_rad)
    return cv2.cvtColor(res_text, cv2.COLOR_BGR2RGB)
