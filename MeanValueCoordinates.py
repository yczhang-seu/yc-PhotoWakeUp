# the implementation of Mean Value Coordinates formula
import math


def dist2d(point1, point2):
    result = math.sqrt(math.pow(point1[0] - point2[0], 2) + math.pow(point1[1] - point2[1], 2))
    if result == 0:
        return 1

    else:
        return result


def tan_halfalpha(cos):

    if cos >= 1:
        return 0

    elif cos <= -1:
        return 20000  # when cos=-0.9999

    else:
        return math.sqrt((1 - cos)/(1 + cos))


def MeanValueCoordinates(contour_vertices, inner_point):

    num_vert = contour_vertices.shape[0]
    cos_value = []

    for i in range(num_vert):
        if i == num_vert - 1:
            a = dist2d(contour_vertices[0], inner_point)
            c = dist2d(contour_vertices[i], contour_vertices[0])

        else:
            a = dist2d(contour_vertices[i + 1], inner_point)
            c = dist2d(contour_vertices[i], contour_vertices[i + 1])

        b = dist2d(contour_vertices[i], inner_point)

        cos = (math.pow(a, 2) + math.pow(b, 2) - math.pow(c, 2))/(2*a*b)
        cos_value.append(cos)

    w_value = []

    for i in range(num_vert):
        if i == 0:
            w = (tan_halfalpha(cos_value[num_vert-1]) + tan_halfalpha(cos_value[i])) / dist2d(contour_vertices[i],
                                                                                              inner_point)

        else:
            w = (tan_halfalpha(cos_value[i-1]) + tan_halfalpha(cos_value[i])) / dist2d(contour_vertices[i],
                                                                                       inner_point)

        w_value.append(w)

    w_total = 0
    lambda_value = []

    for i in range(num_vert):
        w_total += w_value[i]

    for i in range(num_vert):
        lambda_v = w_value[i] / w_total
        lambda_value.append(lambda_v)

    return lambda_value


def main():

    pass


if __name__ == '__main__':
    main()
