# draw label map of smpl model

import numpy as np
import cv2
from warp import getinnerpts


def labelmapfilling(inner_pts, coarse_map, output, iter_=5):

    coarse = cv2.imread(coarse_map, -1)
    filled = coarse.copy()
    inner_list = inner_pts.tolist()

    white = np.array([255, 255, 255])
    black = np.array([0, 0, 0])

    for iteration in range(iter_):

        for inner in range(inner_pts.shape[0]):
            x, y = inner_pts[inner_pts.shape[0] - inner - 1, :]
            if (filled[y, x] == white).all() == True or (filled[y, x] == black).all() == True:
                label1 = np.array([0, 0, 0])
                label2 = np.array([0, 0, 0])
                count1 = 0
                count2 = 0
                init_flag1 = False
                init_flag2 = False
                for i, j in zip([-1, -1, -1, 0, 1, 1, 1, 0], [-1, 0, 1, 1, 1, 0, -1, -1]):
                    if [x + i, y + j] in inner_list and (filled[y + j, x + i] == white).all() == False:
                        if (filled[y + j, x + i] == label1).all() == False and init_flag1 is False:
                            label1 = filled[y + j, x + i]
                            init_flag1 = True
                            count1 += 1
                        elif (filled[y + j, x + i] == label1).all() == False and init_flag1 is True and init_flag2 is False:
                            label2 = filled[y + j, x + i]
                            init_flag2 = True
                            count2 += 1
                        elif (filled[y + j, x + i] == label1).all() == True:
                            count1 += 1
                        elif (filled[y + j, x + i] == label2).all() == True:
                            count2 += 1

                if max(count1, count2) < 2:
                    continue
                elif count1 >= count2:
                    filled[y, x] = label1
                elif count1 < count2:
                    filled[y, x] = label2

            print('%d/%d label map filling' % (inner, inner_pts.shape[0]))
        # print('%d/%d iter' %(iteration, iter_))

    cv2.imshow('label map filled', filled)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite(output, filled)


def main():

    mask_smpl = './data/img_smpl_mask.png'
    pred_vert = './savetxt/pred_vertices.txt'
    head = './smpl_seg_raw/refined/head.txt'
    torso = './smpl_seg_raw/refined/torso.txt'
    left_up = './smpl_seg_raw/refined/left_arm_hand.txt'
    left_down = './smpl_seg_raw/refined/left_leg_foot.txt'
    right_up = './smpl_seg_raw/refined/right_arm_hand.txt'
    right_down = './smpl_seg_raw/refined/right_leg_foot.txt'
    head_index = np.loadtxt(head, dtype=np.int)
    torso_index = np.loadtxt(torso, dtype=np.int)
    left_up_index = np.loadtxt(left_up, dtype=np.int)
    left_down_index = np.loadtxt(left_down, dtype=np.int)
    right_up_index = np.loadtxt(right_up, dtype=np.int)
    right_down_index = np.loadtxt(right_down, dtype=np.int)

    inner_pts = getinnerpts(mask_smpl)
    rect1 = cv2.minAreaRect(inner_pts)

    pred_v = np.loadtxt(pred_vert, dtype=np.float32)
    pred_v2d = pred_v[:, :2] + 1  # to eliminate negative values
    rect2 = cv2.minAreaRect(pred_v2d)

    a = rect1[1][0] / rect2[1][0]
    b = rect1[1][1] / rect2[1][1]
    c = rect1[0][0] - rect2[0][0]
    d = rect1[0][1] - rect2[0][1]

    print(a, b, c, d)
    pred_v2d[:, 0] = (pred_v2d[:, 0] - rect2[0][0]) * a + c
    pred_v2d[:, 1] = (pred_v2d[:, 1] - rect2[0][1]) * b + d
    pred_v2d = np.array(pred_v2d, dtype=np.int)

    mask = cv2.imread(mask_smpl)
    # mask = np.zeros((mask.shape[0], mask.shape[1]))

    # for i in range(pred_v2d.shape[0]):
    #     x, y = pred_v2d[i, :]
    #     cv2.circle(mask, (x, y), 1, (255, 0, 0))

    for i in range(head_index.shape[0]):
        x, y = pred_v2d[head_index[i], :]
        cv2.circle(mask, (x, y), 1, (193, 182, 255))

    for i in range(torso_index.shape[0]):
        x, y = pred_v2d[torso_index[i], :]
        cv2.circle(mask, (x, y), 1, (144, 128, 112))

    for i in range(left_down_index.shape[0]):
        x, y = pred_v2d[left_down_index[i], :]
        cv2.circle(mask, (x, y), 1, (0, 255, 127))

    for i in range(right_up_index.shape[0]):
        x, y = pred_v2d[right_up_index[i], :]
        cv2.circle(mask, (x, y), 1, (0, 69, 255))

    for i in range(right_down_index.shape[0]):
        x, y = pred_v2d[right_down_index[i], :]
        cv2.circle(mask, (x, y), 1, (205, 250, 255))

    for i in range(left_up_index.shape[0]):
        x, y = pred_v2d[left_up_index[i], :]
        cv2.circle(mask, (x, y), 1, (139, 139, 0))

    cv2.imshow('label', mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imwrite('./data/labelmap_coarse.png', mask)

    labelmapfilling(inner_pts, './data/labelmap_coarse.png', './data/labelmap_filled.png')


if __name__ == '__main__':
    main()
