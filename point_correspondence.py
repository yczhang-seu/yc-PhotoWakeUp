# calculate correspondence function of points on the input and smpl silhouettes
import numpy as np
import cv2
import math


def dist2d(point1, point2):
    return math.sqrt(math.pow(point1[0] - point2[0], 2) + math.pow(point1[1] - point2[1], 2))


def getContour(image_path, mode=0):

    mask = cv2.imread(image_path)
    mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    ret, binary = cv2.threshold(mask_gray, 127, 255, cv2.THRESH_BINARY)

    if mode == 0:
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    elif mode == 1:
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    return contours


def dispCorres(img_size, contour1, contour2, phi):

    disp = np.zeros((img_size[0], img_size[1], 3), dtype=np.uint8)
    cv2.drawContours(disp, contour1, -1, (0, 255, 0), 1)  # green
    cv2.drawContours(disp, contour2, -1, (255, 0, 0), 1)  # blue

    contour1 = np.array(contour1)
    contour2 = np.array(contour2)

    len = contour1.shape[1]
    for i in range(0, len, 10):  # do not show all the points when display
        cv2.circle(disp, (contour1[0, i, 0, 0], contour1[0, i, 0, 1]), 1, (255, 0, 0), -1)
        corresPoint = contour2[0, phi[i], 0]
        cv2.circle(disp, (corresPoint[0], corresPoint[1]), 1, (0, 255, 0), -1)
        cv2.line(disp, (contour1[0, i, 0, 0], contour1[0, i, 0, 1]), (corresPoint[0], corresPoint[1]),
                 (255, 255, 255), 1)

    cv2.imshow('point correspondence', disp)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# boundary matching using dynamic programming
# for each point in contour1, match a point in contour2
def dpBoundarymatch(contours1, contours2, k):

    contours1 = np.array(contours1)
    contours2 = np.array(contours2)

    contour1 = np.zeros((contours1[0].shape[0], 2), dtype=np.int)
    for idx in range(contours1[0].shape[0]):
        contour1[idx][0] = contours1[0][idx][0][0]
        contour1[idx][1] = contours1[0][idx][0][1]

    contour2 = np.zeros((contours2[0].shape[0], 2), dtype=np.int)
    for idx in range(contours2[0].shape[0]):
        contour2[idx][0] = contours2[0][idx][0][0]
        contour2[idx][1] = contours2[0][idx][0][1]

    len1 = contour1.shape[0]
    len2 = contour2.shape[0]
    dpValue = np.zeros((len1, len2))
    dpMatch = np.zeros((len1, len2), dtype=np.int)

    dpValue = dpValue + float('+inf')

    for j in range(len2):
        dpValue[0][j] = dist2d(contour1[0], contour2[j])
        dpMatch[0][j] = j

    tmp_dpValue = np.zeros(k+1)

    for i in range(1, len1):
        for j in range(0, len2):
            lastpoint_idx = dpMatch[i-1][j]
            for m in range(0, k+1):
                if lastpoint_idx + m >= len2:
                    lastpoint_idx_mod = lastpoint_idx - len2
                else:
                    lastpoint_idx_mod = lastpoint_idx

                tmp_dpValue[m] = dist2d(contour1[i], contour2[lastpoint_idx_mod + m])

            dpValue[i][j] = dpValue[i-1][j] + np.min(tmp_dpValue)
            dpMatch[i][j] = lastpoint_idx + np.where(tmp_dpValue == np.min(tmp_dpValue))[0][0]
            if dpMatch[i][j] >= len2:
                dpMatch[i][j] -= len2

        print('%d/%d dp bound match' % (i, len1))

    min_j = np.where(dpValue[len1-1] == np.min(dpValue[len1-1]))
    print('the best correlation is: %d' % min_j[0][0])
    phi = dpMatch[:, min_j]
    phi = phi[:, 0, 0]

    return phi


def main():

    pass


if __name__ == '__main__':
    main()
