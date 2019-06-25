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


def process_params(lane_params: dict):
    w_px = lane_params["width_px"] if "width_px" in lane_params else STANDARD_LANE_PARAMS["width_px"]
    l_px = lane_params["length_px"] if "length_px" in lane_params else STANDARD_LANE_PARAMS["length_px"]
    w_met = lane_params["width_m"] if "width_m" in lane_params else STANDARD_LANE_PARAMS["width_m"]
    l_met = lane_params["length_m"] if "length_m" in lane_params else STANDARD_LANE_PARAMS["length_m"]
    return w_px, l_px, w_met, l_met


class Line:
    def __init__(self, fit_coeffs, x_vals, y_vals):
        # x values of the last n fits of the line
        self.recent_x_fitted = [x_vals]
        self.recent_y_fitted = [y_vals]

        # average x values of the fitted line over the last n iterations
        self.bestx = np.mean(x_vals)

        # polynomial coefficients for the most recent fit
        self.current_fit = fit_coeffs

        # radius of curvature of the line in some units
        self.radius_of_curvature = self.measure_curvature_pix(y_vals, fit_coeffs)

    @staticmethod
    def measure_curvature_pix(y, fit_coeffs):
        return ((1. + (2. * fit_coeffs[0] * np.max(y) + fit_coeffs[1]) ** 2) ** 1.5) / (
            np.absolute(2. * fit_coeffs[0]))


class Lane:
    def __init__(self, origin, warped_img, transpose_mtx, transpose_mtx_inv, left_line: Line, right_line: Line,
                 prev_left: Line, prev_right: Line, lane_params):
        standard_lane_width_pix, standard_lane_length_pix, \
        standard_lane_width_m, standard_lane_length_m = process_params(lane_params)

        self.standard_lane_width_m = standard_lane_width_m
        self.standard_lane_length_m = standard_lane_length_m
        self.standard_lane_length_pix = standard_lane_length_pix
        self.standard_lane_width_pix = standard_lane_width_pix
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
        self.lane_width_m = det.calc_lane_width(warped_img.shape[0], left_line.current_fit, right_line.current_fit,
                                                lane_width_m=self.standard_lane_width_m,
                                                lane_width_pix=self.standard_lane_width_pix)
        self.detected = self.sanity_check(warped_img, left_line, right_line, prev_left, prev_right)
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

    def sanity_check(self, warped_img, left_line: Line, right_line: Line, prev_left: Line, prev_right: Line,
                     delta_lane_w=0.5, delta_curv=300, delta_x_pos=50, min_curv=300., max_curv=3000.):
        lane_w_diff = np.absolute(self.standard_lane_width_m - self.lane_width_m)
        lane_w_ok = lane_w_diff <= delta_lane_w

        parallel = det.check_if_parallel(warped_img.shape[0], left_line.current_fit, right_line.current_fit)

        prev_curvature = None if prev_right is None or prev_left is None else self.measure_curvature(
            left_x=prev_left.recent_x_fitted[-1], left_y=prev_left.recent_y_fitted[-1],
            right_x=prev_right.recent_x_fitted[-1], right_y=prev_right.recent_y_fitted[-1])
        curv_ok = (prev_curvature is None or np.absolute(self.radius_of_curvature_m - prev_curvature) <= delta_curv) \
                  and min_curv <= self.radius_of_curvature_m <= max_curv

        left_x_shift = 0 if prev_left is None else np.absolute(left_line.bestx - prev_left.bestx)
        right_x_shift = 0 if prev_right is None else np.absolute(right_line.bestx - prev_right.bestx)
        lines_x_ok = left_x_shift <= delta_x_pos and right_x_shift <= delta_x_pos

        return lane_w_ok and parallel and curv_ok and lines_x_ok

    def draw(self):
        res = vs.draw_lane(self.origin_img, self.binary_warped, self.M_inv,
                           self.left.current_fit, self.right.current_fit)
        dir = vs.Direction.RIGHT if self.vehicle_shift_m >= 0 else vs.Direction.LEFT
        res = vs.display_pos_and_curvature(res, np.absolute(self.vehicle_shift_m), dir, self.radius_of_curvature_m)
        return res


def find_road_lane(img_path, stand_lane_params=STANDARD_LANE_PARAMS):
    img = mpimg.imread(img_path)
    res = find_lane(img, None, stand_lane_params).draw()
    return cv2.cvtColor(res, cv2.COLOR_BGR2RGB)


def find_lane(img, prev_lane: Lane, standard_lane_params=STANDARD_LANE_PARAMS):
    mtx, dst = cam.load_calibration_results()
    undistort = cam.undistort(img, mtx, dst, True)
    process = pr.prepare_for_thresholding(img)
    binary = pr.threshold(process)
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
    prev_left = None if prev_lane is None else prev_lane.left
    prev_right = None if prev_lane is None else prev_lane.right
    lane = Lane(undistort, binary_warped, M, Minv, left_line, right_line, prev_left, prev_right, standard_lane_params)

    return lane


def find_lane_on_video(video_path, out_path='output_videos', output_name='output',
                       standard_lane_params=STANDARD_LANE_PARAMS):
    cap = cv2.VideoCapture(video_path)

    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter(out_path + '/' + output_name + '.mp4', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    lane = None
    step = 0
    last_success_step = 0
    wait_steps = 25

    while cap.isOpened():
        ret, frame = cap.read()

        if ret:
            step += 1
            new_lane = find_lane(frame, lane, standard_lane_params)
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
