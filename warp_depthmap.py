import numpy as np
import cv2
from holefilling import isvalidpix


def isBound(image, x, y):
    flag = False
    white = 0
    # check for 8-neighbor
    for i in range(0, 3):
        for j in range(0, 3):

            if i == 1 and j == 1:
                continue

            elif isvalidpix(image, x - 1 + j, y - 1 + j) == 0:
                flag = True

            elif isvalidpix(image, x - 1 + j, y - 1 + j) == 255:
                white = white + 1

    if flag is True or white >= 3:
        return 1
    else:
        return 0


def valid_neighbor(image, x, y):

    x_valid, y_valid = 0, 0

    if isvalidpix(image, x - 1, y) == 1:
        x_valid = x_valid + 1

    if isvalidpix(image, x + 1, y) == 1:
        x_valid = x_valid + 1

    if isvalidpix(image, x, y - 1) == 1:
        y_valid = y_valid + 1

    if isvalidpix(image, x, y + 1) == 1:
        y_valid = y_valid + 1

    return x_valid, y_valid


def normal_decode(image, x, y):
    x_axis = image[y, x, 2] * 2 - 255  # channel order of opencv imread is BGR
    y_axis = image[y, x, 1] * 2 - 255
    z_axis = image[y, x, 0] * 2 - 255

    return x_axis, y_axis, z_axis


def solve_depth(normal_map_path, depth_map_path, x, y):

    normal_map = cv2.imread(normal_map_path, -1)
    depth_map = cv2.imread(depth_map_path, -1)

    x_axis, y_axis, z_axis = normal_decode(normal_map, x, y)

    if isvalidpix(depth_map, x - 1, y) == 'inner point filled':
        b1 = depth_map[y, x - 1, 0]  # the three channels are the same
    else:
        b1 = depth_map[y, x + 1, 0]

    if isvalidpix(depth_map, x, y - 1) == 'inner point filled':
        b2 = depth_map[y - 1, x, 0]
    else:
        b2 = depth_map[y + 1, x, 0]

    A = [[z_axis], [z_axis]]
    # print(np.array(A))
    b1 = z_axis * b1 - x_axis
    b2 = z_axis * b2 - y_axis
    b = [b1, b2]
    # print(b)

    depth = np.linalg.solve(np.dot(np.array(A).T.copy(), np.array(A)), np.dot(np.array(A).T.copy(), np.array(b)))
    depth = int(depth)

    return depth


def warp_depthmap(depth_map_coarse_path, normal_map_path, inner_pts):

    depth_map_coarse = cv2.imread(depth_map_coarse_path, -1)
    normal_map = cv2.imread(normal_map_path, -1)

    pts_num = inner_pts.shape[0]
    h, w = depth_map_coarse.shape[:2]
    depth_map_filled = np.zeros((h, w, 3), dtype=np.uint8)

    for i in range(0, pts_num):
        x, y = inner_pts[i, :]

        if isBound(depth_map_coarse, x, y) == 1 and depth_map_coarse[y, x, 0] != 255:
            for j in range(0, 3):
                depth_map_filled[y, x, j] = depth_map_coarse[y, x, j]

    cv2.imshow('depth bound', depth_map_filled)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    iter_num = 100
    for iter_ in range(0, iter_num):
        add_count = 0
        for i in range(0, pts_num):
            x, y = inner_pts[i, :]

            if depth_map_filled[y, x, 0] == 0:
                x_valid, y_valid = valid_neighbor(depth_map_filled, x, y)
                if x_valid >= 1 and y_valid >= 1:
                    depth = solve_depth(normal_map, depth_map_filled, x, y)
                    add_count += 1
                    for j in range(0, 3):
                        depth_map_filled[y, x, j] = depth

                else:
                    continue

        print('%d/%d fill depth map, add %d' % (iter_, iter_num, add_count))
        # cv2.imshow('depth', depth_map_filled)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    return depth_map_filled
