# create and refine body label map

import numpy as np
import cv2
import maxflow as mf
from warp import getinnerpts
from labelmap_smpl import labelmapfilling
from sklearn.neighbors import NearestNeighbors

head_label = [193, 182, 255]
torso_label = [144, 128, 112]
left_up_label = [139, 139, 0]
left_down_label = [0, 255, 127]
right_up_label = [0, 69, 255]
right_down_label = [205, 250, 255]

max_value = 9999


def U_p(p_x, p_y, label_points):

    location = np.array([p_x, p_y])
    X = np.vstack((location, label_points))
    nbrs = NearestNeighbors(n_neighbors=2, algorithm='auto').fit(X)
    distances = nbrs.kneighbors([X[0, :]], return_distance=True)

    return distances[0][0][1]


def V_p_q(isSameLabel, gamma=2):

    if isSameLabel:
        return 0

    else:
        return gamma*1


def get_dict(inner_pts):
    map = {}
    invmap = {}

    map_index = 0
    for i in range(inner_pts.shape[0]):
        x, y = inner_pts[i, :]
        map[map_index] = (x, y)
        invmap[(x, y)] = map_index
        map_index += 1

    return map, invmap


def convert_binary_graph(inner_pts, labelmap_path, label):
    labelmap = cv2.imread(labelmap_path, -1)
    binary_map = {}

    for i in range(inner_pts.shape[0]):
        x, y = inner_pts[i, :]
        if (labelmap[y, x] == label).all():
            binary_map[(x, y)] = 1
        else:
            binary_map[(x, y)] = 0

    return binary_map


def update_binary_graph(inner_pts, res):

    binary_map = {}
    for i in range(inner_pts.shape[0]):
        x, y = inner_pts[i, :]
        if res[i] == 1:
            binary_map[(x, y)] = 1
        else:
            binary_map[(x, y)] = 0

    return binary_map


def a_expansion_Graph(inner_pts, binary_map, label_points, not_label_points, disp, part_label):

    mapping, invmap = get_dict(inner_pts)

    Graph = mf.Graph[float](inner_pts.shape[0])
    nodes = Graph.add_nodes(2*inner_pts.shape[0])

    a_index = inner_pts.shape[0]

    # assign n-link edges and t-link of a
    for i in range(len(mapping)):
        x, y = mapping[i]
        for a, b in zip([-1, -1, -1, 0, 1, 1, 1, 0], [-1, 0, 1, 1, 1, 0, -1, -1]):
            if (x + a, y + b) in invmap:
                label_p = binary_map[(x, y)]
                label_q = binary_map[(x + a, y + b)]
                if label_p == label_q:
                    e_weight = 0
                    if label_p == 1:
                        e_weight = V_p_q(True)
                    if label_p == 0:
                        e_weight = V_p_q(False)

                    Graph.add_edge(i, invmap[(x + a, y + b)], e_weight, 0)

                if label_p != label_q:
                    if label_p == 1:
                        e_weight_p_a = V_p_q(True)
                        Graph.add_edge(i, a_index, e_weight_p_a, 0)
                    if label_p == 0:
                        e_weight_p_a = V_p_q(False)
                        Graph.add_edge(i, a_index, e_weight_p_a, 0)
                    if label_q == 1:
                        e_weight_a_q = V_p_q(True)
                        Graph.add_edge(a_index, invmap[(x + a, y + b)], e_weight_a_q, 0)
                    if label_q == 0:
                        e_weight_a_q = V_p_q(False)
                        Graph.add_edge(a_index, invmap[(x + a, y + b)], e_weight_a_q, 0)

                    Graph.add_tedge(a_index, max_value, V_p_q(False))
                    a_index += 1

    # assign t-link
    for i in range(len(mapping)):
        x, y = mapping[i]
        label = binary_map[(x, y)]
        if label == 1:
            Graph.add_tedge(i, U_p(x, y, label_points), max_value)
        if label == 0:
            Graph.add_tedge(i, U_p(x, y, label_points), U_p(x, y, not_label_points))

        print('%d/%d' % (i, len(mapping)))

    flow = Graph.maxflow()
    res = [Graph.get_segment(nodes[i]) for i in range(0, len(nodes))]

    # print(res)
    res = np.array(res)
    # np.savetxt('./savetxt/res.txt', res, fmt='%d')

    label_count = 0
    for i in range(len(mapping)):
        x, y = mapping[i]
        if res[i] == 1:
            disp[y, x] = part_label
            label_count += 1

    print(label_count)
    # cv2.imshow('disp', disp)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return res


def warp_label_map(label_inner, label_map, mask_inner, part_label, disp):
    label_points = []
    not_label_points = []
    for i in range(label_inner.shape[0]):
        x, y = label_inner[i, :]
        if (label_map[y, x] == part_label).all():
            label_points.append([x, y])
        else:
            not_label_points.append([x, y])

    label_points = np.array(label_points)
    print(label_points.shape[0])

    not_label_points = np.array(not_label_points)
    binary_map = convert_binary_graph(mask_inner, './data/img_input_mask.png', part_label)

    for it in range(1):  # no need to iterate
        res = a_expansion_Graph(mask_inner, binary_map, label_points, not_label_points, disp, part_label)
        binary_map = update_binary_graph(mask_inner, res)


def main():

    label_map = cv2.imread('./data/labelmap_filled.png', -1)
    label_inner = getinnerpts('./data/labelmap_filled.png')
    mask_inner = getinnerpts('./data/img_input_mask.png')
    input_mask = cv2.imread('./data/img_input_mask.png')
    disp = input_mask.copy()
    # warp_label_map(label_inner, label_map, mask_inner, head_label, disp)
    # warp_label_map(label_inner, label_map, mask_inner, torso_label, disp)
    # warp_label_map(label_inner, label_map, mask_inner, left_down_label, disp)
    # warp_label_map(label_inner, label_map, mask_inner, right_up_label, disp)
    # warp_label_map(label_inner, label_map, mask_inner, right_down_label, disp)
    # warp_label_map(label_inner, label_map, mask_inner, left_up_label, disp)
    #
    # cv2.imwrite('./data/labelmap_warp_coarse.png', disp)
    labelmapfilling(mask_inner, './data/labelmap_warp_coarse.png', './data/labelmap_warp_filled.png')


if __name__ == '__main__':
    main()
