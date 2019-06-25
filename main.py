import numpy as np
import matplotlib.image as mpimg
import camera as cam
import processing as pr
import detection as det
import visualization as vs
import cv2

STANDARD_LANE_PARAMS = {
    "width_px": 700,
    "length_px": 720,
    "width_m": 3.2,
    "length_m": 20.
}


class Line:
    def __init__(self, fit_coeffs, x_vals, y_vals):

        # x values of the last n fits of the line
        self.recent_x_fitted = [x_vals]
        self.recent_y_fitted = [y_vals]

        # average x values of the fitted line over the last n iterations
        self.bestx = [np.mean(x_vals)]

        # polynomial coefficients averaged over the last n iterations
        self.best_fit = None

        # polynomial coefficients for the most recent fit
        self.current_fit = fit_coeffs

        # radius of curvature of the line in some units
        self.radius_of_curvature = self.measure_curvature_pix(y_vals, fit_coeffs)

        # distance in meters of vehicle center from the line
        self.line_base_pos = None

        # difference in fit coefficients between last and new fits
        self.diffs = np.array([0, 0, 0], dtype='float')

        # x values for detected line pixels
        self.allx = None

        # y values for detected line pixels
        self.ally = None

    def add_fit(self, fit_coeffs, x_vals, y_vals):
        self.diffs = np.absolute(self.current_fit - fit_coeffs)
        self.current_fit = fit_coeffs
        self.recent_x_fitted.append(x_vals)
        self.recent_y_fitted.append(y_vals)
        self.bestx.append(np.mean(x_vals))
        self.radius_of_curvature = self.measure_curvature_pix(y_vals, fit_coeffs)

    @staticmethod
    def measure_curvature_pix(y, fit_coeffs):
        return ((1. + (2. * fit_coeffs[0] * np.max(y) + fit_coeffs[1]) ** 2) ** 1.5) / (
            np.absolute(2. * fit_coeffs[0]))

    def check_similar_curvature(self, new_fit, new_y, coeff_deltas=[0.1, 0.1, 0.1], curve_rad_delta=100):
        similar_fit = True
        for old, new, delta in zip(self.current_fit, new_fit, coeff_deltas):
            if np.absolute(old - new) > delta:
                similar_fit = False

        new_curverad = self.measure_curvature_pix(new_y, new_fit)
        return np.absolute(self.radius_of_curvature - new_curverad) < curve_rad_delta


class Lane:
    def __init__(self, origin, warped_img, transpose_mtx, transpose_mtx_inv, left_line: Line, right_line: Line,
                 prev_curv):
        self.standard_lane_width_m = 3.7
        self.standard_lane_length_m = 20.
        self.standard_lane_length_pix = 720
        self.standard_lane_width_pix = 700
        self.ym_per_pix = self.standard_lane_length_m / self.standard_lane_length_pix
        self.xm_per_pix = self.standard_lane_width_m / self.standard_lane_width_pix

        self.origin_img = origin
        self.binary_warped = warped_img
        self.M = transpose_mtx
        self.M_inv = transpose_mtx_inv

        self.left = left_line
        self.right = right_line
        self.radius_of_curvature_m = self.measure_curvature(
            left_x=self.left.recent_x_fitted[-1], left_y=self.left.recent_y_fitted[-1],
            right_x=self.right.recent_x_fitted[-1], right_y=self.right.recent_y_fitted[-1])
        self.detected = self.sanity_check(warped_img, left_line, right_line, prev_curv)
        self.vehicle_shift_m = self.calc_vehicle_shift_m()

    def measure_curvature(self, left_x, left_y, right_x, right_y):
        left_fit = np.polyfit(left_y * self.ym_per_pix, left_x * self.xm_per_pix, 2)
        right_fit = np.polyfit(right_y * self.ym_per_pix, right_x * self.xm_per_pix, 2)

        left_curverad = ((1. + (2. * left_fit[0] * np.max(left_y) *
                                self.ym_per_pix + left_fit[1]) ** 2) ** 1.5) / (np.absolute(2. * left_fit[0]))
        right_curverad = ((1. + (2. * right_fit[0] * np.max(right_y) *
                                 self.ym_per_pix + right_fit[1]) ** 2) ** 1.5) / (np.absolute(2. * right_fit[0]))

        return (left_curverad + right_curverad) / 2.

    def calc_vehicle_shift_m(self):
        y_eval = np.max(self.left.recent_y_fitted[-1])
        left_line = self.left.current_fit[0] * y_eval ** 2 + self.left.current_fit[1] * y_eval + \
                    self.left.current_fit[2]
        right_line = self.right.current_fit[0] * y_eval ** 2 + self.right.current_fit[1] * y_eval + \
                     self.right.current_fit[2]

        center_of_lane = (right_line - left_line) / 2 + left_line

        dist_left = center_of_lane
        dist_right = self.binary_warped.shape[1] - center_of_lane
        return (dist_right - dist_left) * self.xm_per_pix

    def sanity_check(self, warped_img, left_line: Line, right_line: Line, prev_curvature, delta_lane_w=0.7,
                     delta_curv=300):
        lane_w_diff = np.absolute(self.standard_lane_width_m -
                                  det.calc_lane_width(warped_img.shape[0], left_line.current_fit,
                                                      right_line.current_fit,
                                                      lane_width_m=self.standard_lane_width_m,
                                                      lane_width_pix=self.standard_lane_width_pix))
        lane_w_ok = lane_w_diff <= delta_lane_w

        parallel = det.check_if_parallel(warped_img.shape[0], left_line.current_fit, right_line.current_fit)

        curv_ok = prev_curvature is None or np.absolute(self.radius_of_curvature_m - prev_curvature) <= delta_curv

        return lane_w_ok and parallel and curv_ok

    def draw(self):
        res = vs.draw_lane(self.origin_img, self.binary_warped, self.M_inv,
                           self.left.current_fit, self.right.current_fit)
        coeffs = "curv {} OK {}".format(self.radius_of_curvature_m, self.detected)
        # res = vs.display_text(res, "GOOD" if self.detected else "BAD")
        res = vs.display_text(res, coeffs)

        # dir = vs.Direction.RIGHT if vehicle_shift_m >= 0 else vs.Direction.LEFT
        # res_text = vs.display_pos_and_curvature(res, np.absolute(vehicle_shift_m), dir, curv_rad)
        return res


def find_road_lane(img_path, stand_lane_params=STANDARD_LANE_PARAMS):
    img = mpimg.imread(img_path)
    res = find_lane(img, None)
    return cv2.cvtColor(res, cv2.COLOR_BGR2RGB)


def find_lane(img, prev_lane: Lane, standard_lane_params=STANDARD_LANE_PARAMS):
    mtx, dst = cam.load_calibration_results()
    undistort = cam.undistort(img, mtx, dst, True)

    binary = pr.threshold(undistort)
    src, dst = pr.load_perspective_points()
    M, Minv, binary_warped = pr.perspective_transform(binary, src, dst)

    if prev_lane is None or not prev_lane.detected:
        leftx, lefty, rightx, righty = det.sliding_window(binary_warped)
    else:
        leftx, lefty, rightx, righty = det.search_around_poly(binary_warped,
                                                              prev_lane.left.current_fit, prev_lane.right.current_fit)

    left_fit, right_fit = det.fit_poly(leftx, lefty, rightx, righty)

    left_line = Line(left_fit, leftx, lefty)
    right_line = Line(right_fit, rightx, righty)
    prev_curv = None if prev_lane is None else prev_lane.radius_of_curvature_m
    lane = Lane(undistort, binary_warped, M, Minv, left_line, right_line, prev_curv)

    # isSane = "GOOD" if sanity_check(binary_warped, left_fit, right_fit) else "BAD"

    # curv_rad = det.measure_curvature(leftx, lefty, righty, rightx)
    # vehicle_shift_m = det.calc_vehicle_shift_m(leftx, rightx, binary_warped.shape[1])
    # res = vs.draw_lane(img, binary_warped, Minv, left_fit, right_fit)
    # res = vs.display_text(res, isSane)
    # dir = vs.Direction.RIGHT if vehicle_shift_m >= 0 else vs.Direction.LEFT
    # res_text = vs.display_pos_and_curvature(res, np.absolute(vehicle_shift_m), dir, curv_rad)
    return lane


def find_lane_on_video(video_path, out_path='output_videos', output_name='output',
                       standard_lane_params=STANDARD_LANE_PARAMS):

    cap = cv2.VideoCapture(video_path)

    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter(out_path + '/' + output_name + '.mp4', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    lane = None
    step = 0
    last_success_step = 0
    wait_steps = 10

    while cap.isOpened():
        ret, frame = cap.read()

        if ret:
            step += 1
            new_lane = find_lane(frame, lane)
            if new_lane.detected:
                last_success_step = step
                lane = new_lane  # success, use this lane for next lane detection
            elif step - last_success_step > wait_steps:
                lane = None  # perform sliding window again, too long from last success
            elif lane is not None:
                # if attempt is not successful, just switch origin and retain previous fit
                lane.origin_img = new_lane.origin_img

            frame = lane.draw() if lane is not None else new_lane.draw()
            out.write(frame)

            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


def capture_frames(path):
    cap = cv2.VideoCapture(path)

    # Used as counter variable
    count = 0

    while cap.isOpened():
        ret, frame = cap.read()

        if ret:
            count += 1
            cv2.imwrite("challenge_frames/frame%d.jpg" % count, frame)

    cap.release()
