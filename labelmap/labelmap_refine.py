# refine the occlusion boundary of initial label map
# the implementation of refined body labeling of the original paper is not completed

import numpy as np
import cv2
import warp as wp
from point_correspondence import dpBoundarymatch
from sklearn.mixture import GaussianMixture as GMM
import math
from point_correspondence import dist2d
import labelmap_initial as lmi
import maxflow as mf

head_label = [193, 182, 255]
torso_label = [144, 128, 112]
left_up_label = [139, 139, 0]
left_down_label = [0, 255, 127]
right_up_label = [0, 69, 255]
right_down_label = [205, 250, 255]

max_value = 9999


def getlabelarea(initial_map_path, smpl_map_path, part_label):
    initial_map = cv2.imread(initial_map_path, -1)
    smpl_map = cv2.imread(smpl_map_path, -1)

    initial_inner = wp.getinnerpts(initial_map_path)
    smpl_inner = wp.getinnerpts(smpl_map_path)

    initial_output = np.zeros((224, 224), dtype=np.uint8)
    smpl_output = np.zeros((224, 224), dtype=np.uint8)

    for i in range(initial_inner.shape[0]):
        x, y = initial_inner[i, :]
        if (initial_map[y, x] == part_label).all() == True:
            initial_output[y, x] = 255

    for i in range(smpl_inner.shape[0]):
        x, y = smpl_inner[i, :]
        if (smpl_map[y, x] == part_label).all() == True:
            smpl_output[y, x] = 255

    return initial_output, smpl_output


def label_match(initial_output, smpl_output):
    initial_contour, hierarchy = cv2.findContours(initial_output, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    smpl_contour, hierarchy = cv2.findContours(smpl_output, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(smpl_contour) > 1:
        smpl_contour = smpl_contour[0]
        smpl_contour = np.array(smpl_contour).reshape(1, -1, 1, 2)

    fai = dpBoundarymatch(smpl_contour, initial_contour, 32)
    initial_inner = []
    for i in range(initial_output.shape[0]):
        for j in range(initial_output.shape[1]):
            if initial_output[i, j] == 255:
                initial_inner.append([j, i])

    initial_inner = np.array(initial_inner)
    weights = wp.calweights(initial_contour, fai, initial_inner)

    warp = []

    for i in range(0, weights.shape[0]):
        warpcoord = wp.calcoordinate(smpl_contour, weights[i])
        warpcoord = np.array(warpcoord)
        warp.append(warpcoord)
        print('%d/%d calculate coordinates' % (i, weights.shape[0]))

    warp = np.array(warp)

    match_map = {}
    for i in range(initial_inner.shape[0]):
        match_map[(initial_inner[i, 0], initial_inner[i, 1])] = (warp[i, 0], warp[i, 1])

    return match_map


def pixel_occlusion(initial_map_path, check_initial, map1, map2, map3, smpl_depthmap_path):

    initial_map = cv2.imread(initial_map_path, -1)
    smpl_depthmap = cv2.imread(smpl_depthmap_path, -1)

    check_contours, hierarchy = cv2.findContours(check_initial, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    check_contours = np.array(check_contours)
    occlusion_pairs = []
    thres = 80
    for i in range(check_contours.shape[1]):
        x, y = check_contours[0, i, 0, :]
        for a, b in zip([-1, 0, 1, 0], [0, 1, 0, -1]):
            if (initial_map[y + b, x + a] == initial_map[y, x]).all() == False:
                check_smpl_pos = map1[(x, y)]
                if (initial_map[y + b, x + a] == torso_label).all() == True:
                    cores_smpl_pos = map2[(x + a, y + b)]
                elif (initial_map[y + b, x + a] == left_down_label).all() == True:
                    cores_smpl_pos = map3[(x + a, y + b)]
                else:
                    continue

                check_depth = smpl_depthmap[check_smpl_pos[1], check_smpl_pos[0]]
                cores_depth = smpl_depthmap[cores_smpl_pos[1], cores_smpl_pos[0]]
                if abs(int(check_depth) - int(cores_depth)) >= thres:
                    occlusion_pairs.append([x, y, x + a, y + b])

    occlusion_pairs = np.array(occlusion_pairs)

    return occlusion_pairs


def dilate_occlusion_mask(occlusion_mask_init_path):

    occlusion_mask_init = cv2.imread(occlusion_mask_init_path, -1)
    occlusion_mask_dilate = occlusion_mask_init.copy()
    x_bias = list(range(-2, 3))
    y_bias = list(range(-2, 3))
    for i in range(occlusion_mask_init.shape[0]):
        for j in range(occlusion_mask_init.shape[1]):
            if (occlusion_mask_init[i, j] == [255, 255, 255]).all() == True:
                for a, b in zip(x_bias, y_bias):
                    occlusion_mask_dilate[i + a, j + b] = [255, 255, 255]

    return occlusion_mask_dilate


def getGMM_data(input_path, labelmap_path, mask_O):

    input = cv2.imread(input_path, -1)
    labelmap = cv2.imread(labelmap_path, -1)

    GMM_data = []
    for i in range(mask_O.shape[0]):
        x, y = mask_O[i, :]
        data = input[y, x].tolist()
        label = labelmap[y, x]
        lb = -1

        if (label == left_up_label).all() == True:
            lb = 0
        elif (label == torso_label).all() == True:
            lb = 100
        elif (label == left_down_label).all() == True:
            lb = 200

        if lb == -1:
            continue

        data.append(lb)
        GMM_data.append(data)

    GMM_data = np.array(GMM_data)

    return GMM_data


def U_L_p(x, y, labelmap, input, gmm, mode=0):

    data = input[y, x].tolist()
    label = labelmap[y, x]
    lb = -1
    if (label == left_up_label).all() == True:
        lb = 0
    elif (label == torso_label).all() == True:
        lb = 1
    elif (label == left_down_label).all() == True:
        lb = 2
    data.append(lb)
    data = np.array(data)

    proba_data = gmm.predict_proba(data)

    # todo
    if mode == 0:
        pass
    preb = -1

    return math.log(preb, 10)


def V_L_p_q(p_x, p_y, q_x, q_y, isSameLabel, input, mask_O, gamma=8):

    if not isSameLabel:

        C_p_q = math.pow(dist2d([p_x, p_y], [q_x, q_y]), -1)
        I_p = input[p_y, p_x]
        I_q = input[q_y, q_x]
        dist = 0
        for i in range(3):
            dist += math.pow(int(I_p[i]) - int(I_q[i]), 2)

        mask_O = mask_O.tolist()
        beta = 0
        count = 0
        for a, b in zip([-1, -1, -1, 0, 1, 1, 1, 0], [-1, 0, 1, 1, 1, 0, -1, -1]):
            if [p_x + a, p_y + b] in mask_O:
                d = 0
                I_n = input[p_y + b, p_x + a]
                for i in range(3):
                    d += math.pow(int(I_p[i]) - int(I_n[i]), 2)

                beta += d
                count += 1

        beta = math.pow(2 * beta / count, -1)

        V = C_p_q * math.exp(-beta * dist)

        return gamma * V

    else:
        return 0


def a_expansion_G(mask_O, binary_map, labelmap, input, part_label, GMM_data, disp):

    mapping, invmap = lmi.get_dict(mask_O)
    gmm = GMM(n_components=3).fit(GMM_data)

    Graph = mf.Graph[float](mask_O.shape[0])
    nodes = Graph.add_nodes(2*mask_O.shape[0])

    a_index = mask_O.shape[0]

    # assign n-link edges and t-link of a
    for i in range((len(mapping))):
        x, y = mapping[i]
        for a, b in zip([-1, -1, -1, 0, 1, 1, 1, 0], [-1, 0, 1, 1, 1, 0, -1, -1]):
            label_p = binary_map[(x, y)]
            label_q = binary_map[(x + a, y + b)]
            e_weight = -1
            if label_p == label_q:
                if label_p == 1:
                    e_weight = V_L_p_q(x, y, x + a, y + b, True, input, mask_O)
                elif label_p == 0:
                    e_weight = V_L_p_q(x, y, x + a, y + b, False, input, mask_O)

                Graph.add_edge(i, invmap[(x + a, y + b)], e_weight, 0)

            if label_p != label_q:
                if label_p == 1:
                    e_weight_p_a = V_L_p_q(x, y, x + a, y + b, True, input, mask_O)
                    Graph.add_edge(i, a_index, e_weight_p_a, 0)
                if label_p == 0:
                    e_weight_p_a = V_L_p_q(x, y, x + a, y + b, False, input, mask_O)
                    Graph.add_edge(i, a_index, e_weight_p_a, 0)
                if label_q == 1:
                    e_weight_a_q = V_L_p_q(x, y, x + a, y + b, True, input, mask_O)
                    Graph.add_edge(a_index, invmap[(x + a, y + b)], e_weight_a_q, 0)
                if label_q == 0:
                    e_weight_a_q = V_L_p_q(x, y, x + a, y + b, False, input, mask_O)
                    Graph.add_edge(a_index, invmap[(x + a, y + b)], e_weight_a_q, 0)

                Graph.add_tedge(a_index, max_value, V_L_p_q(x, y, x + a, y + b, False, input, mask_O))
                a_index += 1

        # assign t-link
    for i in range(len(mapping)):
        x, y = mapping[i]
        label = binary_map[(x, y)]
        if label == 1:
            Graph.add_tedge(i, U_L_p(x, y, labelmap, input, gmm), max_value)
        if label == 0:
            Graph.add_tedge(i,  U_L_p(x, y, labelmap, input, gmm),  U_L_p(x, y, labelmap, input, gmm, mode=1))

    flow = Graph.maxflow()
    res = [Graph.get_segment(nodes[i]) for i in range(0, len(nodes))]

    res = np.array(res)

    label_count = 0
    for i in range(len(mapping)):
        x, y = mapping[i]
        if res[i] == 1:
            disp[y, x] = part_label
            label_count += 1

    print(label_count)

    return res


def conbine_main_parts(labelmap_path, inner_pts, occluded_label):

    labelmap = cv2.imread(labelmap_path)
    for i in range(inner_pts.shape[0]):
        x, y = inner_pts[i, :]
        if (labelmap[y, x] == occluded_label).all() == True:
            labelmap[y, x] = [0, 0, 0]

    labelmap_gray = cv2.cvtColor(labelmap, cv2.COLOR_BGR2GRAY)

    ret, binary = cv2.threshold(labelmap_gray, 2, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = np.array(contours)
    contours = contours[0].reshape(1, -1, 1, 2)
    disp = np.zeros((224, 224), dtype=np.uint8)
    for i in range(contours.shape[1]):
        x, y = contours[0, i, 0, :]
        cv2.circle(disp, (x, y), 1, (255, 255, 255))

    cv2.circle(disp, (132, 110), 10, (255, 255, 0))
    cv2.circle(disp, (123, 68), 5, (255, 255, 0))
    cv2.imshow('disp', disp)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return contours


def find_occlusion_endpoints(contours, mask_O):

    contour_points = []
    for i in range(contours.shape[1]):
        x, y = contours[0, i, 0, :]
        contour_points.append([x, y])

    mask_O = mask_O.tolist()

    flag = False
    count = 0
    for i in range(len(contour_points)):
        if contour_points[i] in mask_O and flag is False:
            start = contour_points[i]
            flag = True
        elif contour_points[i] not in mask_O and flag is True:
            count += 1
            flag = False
            if count == 4:
                end = contour_points[i - 1]
                break

    return contour_points, start, end


def main():

    # left_up_initial, left_up_smpl = getlabelarea('./data/labelmap_warp_filled.png', './data/labelmap_filled.png',
    #                                              left_up_label)
    # left_up_map = label_match(left_up_initial, left_up_smpl)
    #
    # left_down_initial, left_down_smpl = getlabelarea('./data/labelmap_warp_filled.png', './data/labelmap_filled.png',
    #                                                  left_down_label)
    #
    # left_down_map = label_match(left_down_initial, left_down_smpl)
    #
    # torso_initial, torso_smpl = getlabelarea('./data/labelmap_warp_filled.png', './data/labelmap_filled.png',
    #                                          torso_label)
    # torso_map = label_match(torso_initial, torso_smpl)
    #
    # occlusion_pairs = pixel_occlusion('./data/labelmap_warp_filled.png', left_up_initial, left_up_map, torso_map,
    #                                   left_down_map, './data/img_depth.png')
    # occlusion_mask_init = cv2.imread('./data/labelmap_warp_filled.png', -1)
    # for i in range(occlusion_pairs.shape[0]):
    #     x1, y1, x2, y2 = occlusion_pairs[i, :]
    #     occlusion_mask_init[y1, x1] = [255, 255, 255]
    #     occlusion_mask_init[y2, x2] = [255, 255, 255]
    #
    # cv2.imwrite('./data/occlusion_mask_init.png', occlusion_mask_init)

    # occlusion_mask_dilate = dilate_occlusion_mask('./data/occlusion_mask_init.png')
    # cv2.imwrite('./data/occlusion_mask_dilate.png', occlusion_mask_dilate)

    mask_O = []
    occlusion_mask_dilate = cv2.imread('./data/occlusion_mask_dilate.png', -1)
    for i in range(occlusion_mask_dilate.shape[0]):
        for j in range(occlusion_mask_dilate.shape[1]):
            if (occlusion_mask_dilate[i, j] == [255, 255, 255]).all() == True:
                mask_O.append([j, i])

    mask_O = np.array(mask_O)

    label_inner = wp.getinnerpts('./data/labelmap_warp_filled.png')
    contours = conbine_main_parts('./data/labelmap_warp_filled.png', label_inner, left_up_label)
    contour_points, start, end = find_occlusion_endpoints(contours, mask_O)
    print(start, end)


if __name__ == '__main__':
    main()
