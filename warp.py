import numpy as np
import cv2
from MeanValueCoordinates import MeanValueCoordinates as MVC


def calcoordinate(contour, weights):

    contour = np.array(contour)
    num = contour.shape[1]
    x, y = 0, 0
    for i in range(0, num):
        x = x + contour[0, i, 0, 0] * weights[i]
        y = y + contour[0, i, 0, 1] * weights[i]

    coordinate = np.array([x, y], dtype=np.int)

    return coordinate


def getinnerpts(image_path):

    image = cv2.imread(image_path, -1)
    h, w = image.shape[:2]
    c = len(image.shape)

    inner_pts = []

    if c == 3:
        for i in range(0, h):
            for j in range(0, w):
                if image[i, j, 0] != 0 or image[i, j, 1] != 0 or image[i, j, 2] != 0:
                    location = np.array([j, i])
                    inner_pts.append(location)

    if c == 2:
        for i in range(0, h):
            for j in range(0, w):
                if image[i, j] != 0:
                    location = np.array([j, i])
                    inner_pts.append(location)

    inner_pts = np.array(inner_pts)

    return inner_pts


def calweights(contour, phi, inner_pts):

    contour = np.array(contour)
    base = []
    w = []
    for i in range(0, phi.shape[0]):
        base.append(contour[0, phi[i], 0, :])

    base = np.array(base)
    for i in range(0, inner_pts.shape[0]):
        weights = MVC(base, inner_pts[i])
        w.append(weights)

        print('%d/%d calculate weights' % (i, inner_pts.shape[0]))

    w = np.array(w).reshape(-1, phi.shape[0])

    return w


def warpmap(bg_path, map_path, output_path, contour, inner_pts, weights):

    map_warp = cv2.imread(bg_path, 1)  # read background as 3-channel
    map_origin = cv2.imread(map_path, 1)

    warp = []

    for i in range(weights.shape[0]):
        warpcoord = calcoordinate(contour, weights[i])
        warpcoord = np.array(warpcoord)
        warp.append(warpcoord)
        print('%d/%d calculate coordinates' % (i, weights.shape[0]))

    warp = np.array(warp)

    pts_num = inner_pts.shape[0]
    for i in range(pts_num):
        x1, y1 = inner_pts[i, :]
        x2, y2 = warp[i, :]

        for c in range(3):
            map_warp[y2, x2, c] = map_origin[y1, x1, c]

        print('%d/%d warp map' % (i, pts_num))

    cv2.imwrite(output_path, map_warp)


def main():
    pass


if __name__ == '__main__':
    main()
