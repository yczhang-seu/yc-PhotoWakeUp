# extract the predicted vertices of each part

import numpy as np

head = np.loadtxt('./smpl_seg_raw/refined/head.txt', dtype=np.int)
left_up = np.loadtxt('./smpl_seg_raw/refined/left_arm_hand.txt', dtype=np.int)
left_down = np.loadtxt('./smpl_seg_raw/refined/left_leg_foot.txt', dtype=np.int)
right_up = np.loadtxt('./smpl_seg_raw/refined/right_arm_hand.txt', dtype=np.int)
right_down = np.loadtxt('./smpl_seg_raw/refined/right_leg_foot.txt', dtype=np.int)
torso = np.loadtxt('./smpl_seg_raw/refined/torso.txt', dtype=np.int)

pred_vert = np.loadtxt('./savetxt/pred_vertices.txt')

total_index = np.concatenate((head, left_down), axis=0)
total_index = np.concatenate((total_index, right_up), axis=0)
total_index = np.concatenate((total_index, right_down), axis=0)
total_index = np.concatenate((total_index, torso), axis=0)

part_vert = []
for i in range(total_index.shape[0]):
    part_vert.append(pred_vert[total_index[i]])

part_vert = np.array(part_vert)
np.savetxt('./savetxt/part_vert.txt', part_vert, fmt='%.5f')

occluded_vert = []
for i in range(left_up.shape[0]):
    occluded_vert.append(pred_vert[left_up[i]])

occluded_vert = np.array(occluded_vert)
np.savetxt('./savetxt/occluded_vert.txt', occluded_vert, fmt='%.5f')
