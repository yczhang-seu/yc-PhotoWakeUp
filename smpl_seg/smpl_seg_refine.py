# refine the smpl part segmentation from smpl_seg_raw

import numpy as np
import cv2
from sklearn.cluster import DBSCAN


whole_vert = './smpl_seg_raw/whole_body.txt'
pred_vert = './pred_vertices.txt'


def getpartindex(part_vert, whole_vert, pred_vert, output, min=3):

    part = np.loadtxt(part_vert, usecols=(1, 2, 3))
    whole = np.loadtxt(whole_vert, usecols=(1, 2, 3))

    part_index_raw = []
    for i in range(0, part.shape[0]):
        value = part[i, :]
        index = np.argwhere(value == whole)
        part_index_raw.append(index[0, 0])

    part_index_raw = np.array(part_index_raw)
    pred = np.loadtxt(pred_vert)
    vert2d = pred[:, :2]
    vert2d_t = vert2d * 200 + 270  # visualization

    disp1 = np.zeros((500, 500, 3), dtype=np.uint8)
    disp2 = np.zeros((500, 500, 3), dtype=np.uint8)

    point = []
    for i in range(0, part_index_raw.shape[0]):
        cv2.circle(disp1, (int(vert2d_t[part_index_raw[i], 0]),
                           int(vert2d_t[part_index_raw[i], 1])), 1, (255, 255, 255), -1)
        point.append([vert2d_t[part_index_raw[i], 0], vert2d_t[part_index_raw[i], 1]])

    point = np.array(point)

    cv2.imshow('vert2d_raw', disp1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    estimator = DBSCAN(eps=6, min_samples=min).fit(point)
    label = estimator.labels_  # point labeled with -1 is noise

    part_index_refined = []
    noise = []

    for i in range(0, part_index_raw.shape[0]):

        if label[i] != -1:
            cv2.circle(disp2, (int(vert2d_t[part_index_raw[i], 0]),
                               int(vert2d_t[part_index_raw[i], 1])), 1, (255, 255, 255), -1)
            part_index_refined.append(part_index_raw[i])

        else:
            noise.append(i)

    part_index_refined = np.array(part_index_refined)
    noise = np.array(noise)

    cv2.imshow('vert2d_refined', disp2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    np.savetxt(output, part_index_refined, fmt='%d')

    return part_index_refined, noise


def main():

    head, head_noise = getpartindex('./smpl_seg_raw/head.txt',
                                    whole_vert, pred_vert, './smpl_seg_raw/refined/head.txt')
    left_arm, left_arm_noise = getpartindex('./smpl_seg_raw/left_arm.txt',
                                            whole_vert, pred_vert, './smpl_seg_raw/refined/left_arm.txt')
    left_foot, left_foot_noise = getpartindex('./smpl_seg_raw/left_foot.txt',
                                              whole_vert, pred_vert, './smpl_seg_raw/refined/left_foot.txt')
    left_hand, left_hand_noise = getpartindex('./smpl_seg_raw/left_hand.txt',
                                              whole_vert, pred_vert, './smpl_seg_raw/refined/left_hand.txt')
    left_leg, left_leg_noise = getpartindex('./smpl_seg_raw/left_leg.txt',
                                            whole_vert, pred_vert, './smpl_seg_raw/refined/left_leg.txt')
    right_arm, right_arm_noise = getpartindex('./smpl_seg_raw/right_arm.txt',
                                              whole_vert, pred_vert, './smpl_seg_raw/refined/right_arm.txt')
    right_foot, right_foot_noise = getpartindex('./smpl_seg_raw/right_foot.txt',
                                                whole_vert, pred_vert, './smpl_seg_raw/refined/right_foot.txt')
    right_hand, right_hand_noise = getpartindex('./smpl_seg_raw/right_hand.txt',
                                                whole_vert, pred_vert, './smpl_seg_raw/refined/right_hand.txt', min=15)
    right_leg, right_leg_noise = getpartindex('./smpl_seg_raw/right_leg.txt',
                                              whole_vert, pred_vert, './smpl_seg_raw/refined/right_leg.txt')
    torso, torso_noise = getpartindex('./smpl_seg_raw/torso.txt',
                                      whole_vert, pred_vert, './smpl_seg_raw/refined/torso.txt', min=10)

    # print(head_noise.shape[0] + left_arm_noise.shape[0] + left_foot_noise.shape[0] +
    #       left_hand_noise.shape[0] + left_leg_noise.shape[0] + right_arm_noise.shape[0] + right_foot_noise.shape[0] +
    #       right_hand_noise.shape[0] + right_leg_noise.shape[0] + torso_noise.shape[0])


if __name__ == '__main__':
    main()
