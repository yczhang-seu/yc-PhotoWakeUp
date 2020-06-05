# output stitched 3D mesh in obj format
import cv2
import point_correspondence as pc
import warp as wp
import holefilling as hf
import output as op
import stitch as st

# dir settings
mask_input_front = './data/mask_input.png'
mask_smpl_front = './data/mask_smpl.png'
depth_smpl_front = './data/smpl_depth.png'
depth_smpl_back = './data/smpl_depth_back.png'

output_obj = './output/out.obj'

mask_input_back = mask_input_front
mask_smpl_back = mask_smpl_front
depth_coarse_front = depth_smpl_front[0:len(depth_smpl_front)-4] + '_coarse.png'
depth_filled_front = depth_smpl_front[0:len(depth_smpl_front)-4] + '_filled.png'
depth_smpl_back_flip = depth_smpl_back[0:len(depth_smpl_front)-4] + '_flip.png'
depth_coarse_back = depth_smpl_back[0:len(depth_smpl_back)-4] + '_coarse.png'
depth_filled_back = depth_smpl_back[0:len(depth_smpl_back)-4] + '_filled.png'


def warpnfill(mask_input, mask_smpl, map_smpl, warp_coarse, warp_filled, ifoutlier=False):

    contour_input = pc.getContour(mask_input, mode=0)
    contour_smpl = pc.getContour(mask_smpl, mode=0)

    phi = pc.dpBoundarymatch(contour_input, contour_smpl, 32)

    # to check the contour point match
    # pc.dispCorres([224, 224], contour_input, contour_smpl, phi)
    # exit()

    inner_smpl = wp.getinnerpts(mask_smpl)
    weights = wp.calweights(contour_smpl, phi, inner_smpl)

    wp.warpmap(mask_input, map_smpl, warp_coarse, contour_input, inner_smpl, weights)

    inner_input = wp.getinnerpts(mask_input)
    input_contour_all = pc.getContour(mask_input, mode=1)
    hf.holefilling(warp_coarse, warp_filled, inner_input, input_contour_all, mask_input_front, outlier=ifoutlier)


def reconstruct(depth_filled_front, depth_filled_back, output_obj):

    c_front, c_back, i_front, i_back = st.initialcontours(depth_filled_front, depth_filled_back)
    bias = st.mesh_align(i_back, i_front)

    front_idx = 0
    back_idx = 1
    inner_front, vert_dict_front = op.outputvert(i_front, depth_filled_front, output_obj, front_idx)

    inner_back, vert_dict_back = op.outputvert(i_back, depth_filled_back, output_obj, back_idx,
                                               isback=True, bias=bias)

    for key in vert_dict_back:
        vert_dict_back[key] += inner_front.shape[0]

    vert_dict_merge = {**vert_dict_front, **vert_dict_back}

    op.outputmesh(inner_front, vert_dict_merge, output_obj, front_idx)
    op.outputmesh(inner_back, vert_dict_merge, output_obj, back_idx, isback=True)
    
    st.mesh_stitch(c_back, c_front, back_idx, front_idx, vert_dict_merge, output_obj, bias=bias)
    st.mesh_stitch(c_front, c_back, front_idx, back_idx, vert_dict_merge, output_obj)


def main():

    if cv2.imread(depth_smpl_back_flip) is None:
        back = cv2.imread(depth_smpl_back, -1)
        back = cv2.flip(back, 1)
        cv2.imwrite(depth_smpl_back_flip, back)

    warpnfill(mask_input_front, mask_smpl_front, depth_smpl_front, depth_coarse_front, depth_filled_front, ifoutlier=False)
    warpnfill(mask_input_back, mask_smpl_back, depth_smpl_back_flip, depth_coarse_back, depth_filled_back, ifoutlier=True)

    reconstruct(depth_filled_front, depth_filled_back, output_obj)


if __name__ == '__main__':
    main()
