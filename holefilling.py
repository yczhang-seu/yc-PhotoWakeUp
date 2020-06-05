import numpy as np
from MeanValueCoordinates import MeanValueCoordinates as MVC
import cv2
from sklearn.neighbors import NearestNeighbors


def isvalidpix(image, x, y):

    if (image[y, x] == [0, 0, 0]).all() == True:
        return 'not inner point'

    elif (image[y, x] == [255, 255, 255]).all() == True:
        return 'inner point not filled'

    else:
        return 'inner point filled'


def collectneighbor(image, x, y, neighbor, edge_flag):

    if isvalidpix(image, x, y) == 'not inner point':
        edge_flag = True
    elif isvalidpix(image, x, y) == 'inner point filled':
        neighbor.append([x, y])

    return edge_flag


def outlier_elim(filled_map, inner_pts):

    for i in range(inner_pts.shape[0]):
        x, y = inner_pts[i, :]
        sum = 0
        count = 0
        for a in range(-5, 6):
            for b in range(-5, 6):
                if x + a >= 224 or y + b >= 224:
                    continue
                elif isvalidpix(filled_map, x + a, y + b) == 'inner point filled':
                    sum += filled_map[y + b, x + a, 0]
                    count += 1

        average = sum / count
        if filled_map[y, x, 0] <= average / 2:
            for c in range(3):
                filled_map[y, x, c] = average

        print('%d/%d outlier eliminate' % (i, inner_pts.shape[0]))


def holefilling(coarse_warp_img_path, output_path, inner_pts, input_contour_all, input_mask_path, outlier=False):

    coarse_warp_img = cv2.imread(coarse_warp_img_path, -1)
    pts_num = inner_pts.shape[0]
    edge_hole = []
    filled_warp_img = coarse_warp_img.copy()

    for i in range(0, pts_num):
        print('%d/%d hole filling' % (i, pts_num))
        edge_flag = False

        x, y = inner_pts[i, :]

        if isvalidpix(coarse_warp_img, x, y) == 'inner point not filled':

            neighbor_8 = []

            # search for 8-neighbor, in anti-clockwise
            for a, b in zip([-1, -1, -1, 0, 1, 1, 1, 0], [-1, 0, 1, 1, 1, 0, -1, -1]):
                edge_flag = collectneighbor(coarse_warp_img, x + a, y + b, neighbor_8, edge_flag)

            if edge_flag:
                edge_hole.append([x, y])
                continue

            else:

                neighbor_8 = np.array(neighbor_8).reshape(-1, 2)

                if neighbor_8.shape[0] >= 4:
                    weights = MVC(neighbor_8, [x, y])
                    weights = np.array(weights)
                    num_weights = weights.shape[0]

                    filled_warp_img[y, x, 0] = 0
                    filled_warp_img[y, x, 1] = 0
                    filled_warp_img[y, x, 2] = 0

                    for k in range(0, num_weights):
                        filled_warp_img[y, x, 0] += filled_warp_img[neighbor_8[k, 1], neighbor_8[k, 0], 0] * weights[k]
                        filled_warp_img[y, x, 1] += filled_warp_img[neighbor_8[k, 1], neighbor_8[k, 0], 1] * weights[k]
                        filled_warp_img[y, x, 2] += filled_warp_img[neighbor_8[k, 1], neighbor_8[k, 0], 2] * weights[k]

                else:
                    edge_hole.append([x, y])

    edge_hole = np.array(edge_hole).reshape(-1, 2)
    edge_hole_num = edge_hole.shape[0]

    filled_pixel = []
    for i in range(inner_pts.shape[0]):
        x, y = inner_pts[i, :]
        if isvalidpix(filled_warp_img, x, y) == 'inner point filled':
            filled_pixel.append([x, y])

    filled_pixel = np.array(filled_pixel)
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(filled_pixel)

    for i in range(edge_hole_num):
        x, y = edge_hole[i, :]
        idx = nbrs.kneighbors(np.array([x, y]).reshape(-1, 2), return_distance=False)
        x_n, y_n = filled_pixel[idx[0, 0], :]
        for c in range(3):
            filled_warp_img[y, x, c] = filled_warp_img[y_n, x_n, c]

    if outlier is True:
        outlier_elim(filled_warp_img, inner_pts)

    for i in range(inner_pts.shape[0]):
        x, y = inner_pts[i, :]
        sum = 0
        count = 0
        for a in range(-1, 2):
            for b in range(-1, 2):
                if x + a >= 224 or y + b >= 224:
                    continue
                elif isvalidpix(filled_warp_img, x + a, y + b) == 'inner point filled':
                    sum += filled_warp_img[y + b, x + a, 0]
                    count += 1
        if count == 0:
            continue

        average = sum / count
        for c in range(3):
            filled_warp_img[y, x, c] = average

    input_contour_all = np.array(input_contour_all)

    for i in range(input_contour_all.shape[1]):
        x, y = input_contour_all[0, i, 0, :]
        sum = 0
        for j in range(-10, 10):
            if i + j < 0:
                i += input_contour_all.shape[1]
            elif i + j >= input_contour_all.shape[1]:
                i -= input_contour_all.shape[1]

            x_n, y_n = input_contour_all[0, i + j, 0, :]
            sum += filled_warp_img[y_n, x_n, 0]
        average = sum / 20 + (6 - 0.025 * sum / 20)
        for c in range(3):
            filled_warp_img[y, x, c] = average

    input_mask = cv2.imread(input_mask_path, -1)
    input_mask = cv2.cvtColor(input_mask, cv2.COLOR_BGR2GRAY)
    filled_warp_img = cv2.bitwise_and(filled_warp_img, filled_warp_img, mask=input_mask)

    cv2.imwrite(output_path, filled_warp_img)
