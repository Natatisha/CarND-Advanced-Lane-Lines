import numpy as np
import cv2
import pickle
import matplotlib.pyplot as plt
import os


def extract_obj_points(img_path, nx, ny, out_folder='cb_corners'):
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((nx * ny, 3), np.float32)

    objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

    # Step through the list and search for chessboard corners
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    pattern_found, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

    cv2.drawChessboardCorners(img, (nx, ny), corners, pattern_found)
    img_name = os.path.basename(img_path)
    write_name = os.path.join(out_folder, 'corners_found_' + img_name)
    cv2.imwrite(write_name, img)

    return pattern_found, objp, corners


def calibrate_camera(objpoints, imgpoints, img_size):
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

    # Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
    dist_pickle = {}
    dist_pickle["mtx"] = mtx
    dist_pickle["dist"] = dist
    pickle.dump(dist_pickle, open("camera_cal/wide_dist_pickle.p", "wb"))

    return ret, mtx, dist, rvecs, tvecs
