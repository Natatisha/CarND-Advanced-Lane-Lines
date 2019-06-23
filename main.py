import numpy as np
import matplotlib.image as mpimg
import camera as cam
import processing as pr
import detection as det
import visualization as vs
import cv2


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


# class Lane():
#     def __init__(self, leftLine: Line, rightLine: Line):
#         self.detected = sanity_check()


def sanity_check(warped_img, left_fit, right_fit, standard_lane_width_m=3.7, delta_meters=0.7):
    img_h = warped_img.shape[0]

    lane_w = det.calc_lane_width(img_h, left_fit, right_fit)
    lane_w_ok = standard_lane_width_m - delta_meters <= lane_w <= standard_lane_width_m + delta_meters

    return lane_w_ok and det.check_if_parallel(img_h, left_fit, right_fit)


def find_road_lane(img_path):
    img = mpimg.imread(img_path)
    res = find_lane(img)
    return cv2.cvtColor(res, cv2.COLOR_BGR2RGB)


def find_lane(img):
    mtx, dst = cam.load_calibration_results()
    undistort = cam.undistort(img, mtx, dst, True)
    binary = pr.threshold(undistort)
    src, dst = pr.load_perspective_points()
    M, Minv, binary_warped = pr.perspective_transform(binary, src, dst)
    leftx, lefty, rightx, righty = det.sliding_window(binary_warped)
    left_fit, right_fit = det.fit_poly(leftx, lefty, rightx, righty)
    isSane = "GOOD" if sanity_check(binary_warped, left_fit, right_fit) else "BAD"

    # curv_rad = det.measure_curvature(leftx, lefty, righty, rightx)
    # vehicle_shift_m = det.calc_vehicle_shift_m(leftx, rightx, binary_warped.shape[1])
    res = vs.draw_lane(img, binary_warped, Minv, left_fit, right_fit)
    res = vs.display_text(res, isSane)
    # dir = vs.Direction.RIGHT if vehicle_shift_m >= 0 else vs.Direction.LEFT
    # res_text = vs.display_pos_and_curvature(res, np.absolute(vehicle_shift_m), dir, curv_rad)
    return res


def find_lane_on_video(video_path, out_path='output_videos', output_name='output'):
    cap = cv2.VideoCapture(video_path)

    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter(out_path + '/' + output_name + '.mp4', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            mtx, dst = cam.load_calibration_results()
            undistort = cam.undistort(frame, mtx, dst, True)
            src, dst = pr.load_perspective_points()
            M, Minv, binary_warped = pr.perspective_transform(undistort, src, dst)

            frame = find_lane(frame)
            # frame = binary_warped
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
