# output the 3D mesh to obj format
import numpy as np
import cv2


def outputvert(inner_pts, depthmap_path, outfile, mesh_idx, thickness=280, isback=False, bias=0):

    depthmap = cv2.imread(depthmap_path)  # read depth map as 3-channel
    inner_pts = np.array(inner_pts)
    inner_num = inner_pts.shape[0]
    for i in range(inner_num):
        inner_pts[i, 0] += bias

    vert_dict = {}

    with open(outfile, 'a+') as output:

        for i in range(inner_num):
            x, y = inner_pts[i, :]

            vert_dict[(x, y, mesh_idx)] = i + 1

            if not isback:
                z = thickness - depthmap[y][x-bias][0]

            if isback:
                z = depthmap[y][x-bias][0]

            z = z / 4
            x = x - depthmap.shape[1] / 2
            y = y - depthmap.shape[0] / 2

            vert = 'v %.1f %.1f %.2f\n' % (x, y, z)
            output.write(vert)

    return inner_pts, vert_dict


def outputmesh(inner_pts, vert_dict, outfile, mesh_idx, isback=False):

    inner_num = inner_pts.shape[0]
    inner_pts_list = inner_pts.tolist()

    with open(outfile, 'a+') as output:

        for i in range(inner_num):
            x, y = inner_pts[i, :]
            mesh = [(x, y)]

            # search in clockwise
            if [x+1, y] in inner_pts_list:
                mesh.append((x+1, y))

            if [x+1, y+1] in inner_pts_list:
                mesh.append((x+1, y+1))

            if [x, y+1] in inner_pts_list:
                mesh.append((x, y+1))

            mesh = np.array(mesh)
            mesh.reshape(-1, 2)

            if mesh.shape[0] == 4:
                a = vert_dict[(mesh[0, 0], mesh[0, 1], mesh_idx)]
                b = vert_dict[(mesh[1, 0], mesh[1, 1], mesh_idx)]
                c = vert_dict[(mesh[2, 0], mesh[2, 1], mesh_idx)]
                d = vert_dict[(mesh[3, 0], mesh[3, 1], mesh_idx)]

                if not isback:
                    mesh_write = 'f %d %d %d\nf %d %d %d\n' % (a, b, d, b, c, d)

                if isback:
                    mesh_write = 'f %d %d %d\nf %d %d %d\n' % (a, d, b, d, c, b)

                output.write(mesh_write)

            if mesh.shape[0] == 3:
                a = vert_dict[(mesh[0, 0], mesh[0, 1], mesh_idx)]
                b = vert_dict[(mesh[1, 0], mesh[1, 1], mesh_idx)]
                c = vert_dict[(mesh[2, 0], mesh[2, 1], mesh_idx)]

                if not isback:
                    mesh_write = 'f %d %d %d\n' % (a, b, c)

                if isback:
                    mesh_write = 'f %d %d %d\n' % (a, c, b)

                output.write(mesh_write)

            print('%d/%d output mesh' % (i, inner_num))


def main():

    pass


if __name__ == '__main__':
    main()
