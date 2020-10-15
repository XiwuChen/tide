import numpy as np
from scipy.spatial import Delaunay
import scipy
from . import object3d
import torch
import cv2


def get_objects_from_label(label_file):
    with open(label_file, 'r') as f:
        lines = f.readlines()
    objects = [object3d.Object3d(line) for line in lines]
    return objects


def dist_to_plane(plane, points):
    """
    Calculates the signed distance from a 3D plane to each point in a list of points
    :param plane: (a, b, c, d)
    :param points: (N, 3)
    :return: (N), signed distance of each point to the plane
    """
    a, b, c, d = plane

    points = np.array(points)
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    return (a * x + b * y + c * z + d) / np.sqrt(a ** 2 + b ** 2 + c ** 2)


def rotate_pc_along_y(pc, rot_angle):
    """
    params pc: (N, 3+C), (N, 3) is in the rectified camera coordinate
    params rot_angle: rad scalar
    Output pc: updated pc with XYZ rotated
    """
    cosval = np.cos(rot_angle)
    sinval = np.sin(rot_angle)
    rotmat = np.array([[cosval, -sinval], [sinval, cosval]])
    pc[:, [0, 2]] = np.dot(pc[:, [0, 2]], np.transpose(rotmat))
    return pc


def rotate_pc_along_y_torch(pc, rot_angle):
    """
    :param pc: (N, 512, 3 + C)
    :param rot_angle: (N)
    :return:
    TODO: merge with rotate_pc_along_y_torch in bbox_transform.py
    """
    cosa = torch.cos(rot_angle).view(-1, 1)  # (N, 1)
    sina = torch.sin(rot_angle).view(-1, 1)  # (N, 1)

    raw_1 = torch.cat([cosa, -sina], dim = 1)  # (N, 2)
    raw_2 = torch.cat([sina, cosa], dim = 1)  # (N, 2)
    R = torch.cat((raw_1.unsqueeze(dim = 1), raw_2.unsqueeze(dim = 1)), dim = 1)  # (N, 2, 2)

    pc_temp = pc[:, :, [0, 2]]  # (N, 512, 2)

    pc[:, :, [0, 2]] = torch.matmul(pc_temp, R.permute(0, 2, 1))  # (N, 512, 2)

    return pc


def boxes3d_to_corners3d(boxes3d, rotate = True):
    """
    :param boxes3d: (N, 7) [x, y, z, h, w, l, ry]
    :param rotate:
    :return: corners3d: (N, 8, 3)
    """
    boxes_num = boxes3d.shape[0]
    h, w, l = boxes3d[:, 3], boxes3d[:, 4], boxes3d[:, 5]
    x_corners = np.array([l / 2., l / 2., -l / 2., -l / 2., l / 2., l / 2., -l / 2., -l / 2.],
                         dtype = np.float32).T  # (N, 8)
    z_corners = np.array([w / 2., -w / 2., -w / 2., w / 2., w / 2., -w / 2., -w / 2., w / 2.],
                         dtype = np.float32).T  # (N, 8)

    y_corners = np.zeros((boxes_num, 8), dtype = np.float32)
    y_corners[:, 4:8] = -h.reshape(boxes_num, 1).repeat(4, axis = 1)  # (N, 8)

    if rotate:
        ry = boxes3d[:, 6]
        zeros, ones = np.zeros(ry.size, dtype = np.float32), np.ones(ry.size, dtype = np.float32)
        rot_list = np.array([[np.cos(ry), zeros, -np.sin(ry)],
                             [zeros, ones, zeros],
                             [np.sin(ry), zeros, np.cos(ry)]])  # (3, 3, N)
        R_list = np.transpose(rot_list, (2, 0, 1))  # (N, 3, 3)

        temp_corners = np.concatenate((x_corners.reshape(-1, 8, 1), y_corners.reshape(-1, 8, 1),
                                       z_corners.reshape(-1, 8, 1)), axis = 2)  # (N, 8, 3)
        rotated_corners = np.matmul(temp_corners, R_list)  # (N, 8, 3)
        x_corners, y_corners, z_corners = rotated_corners[:, :, 0], rotated_corners[:, :, 1], rotated_corners[:, :, 2]

    x_loc, y_loc, z_loc = boxes3d[:, 0], boxes3d[:, 1], boxes3d[:, 2]

    x = x_loc.reshape(-1, 1) + x_corners.reshape(-1, 8)
    y = y_loc.reshape(-1, 1) + y_corners.reshape(-1, 8)
    z = z_loc.reshape(-1, 1) + z_corners.reshape(-1, 8)

    corners = np.concatenate((x.reshape(-1, 8, 1), y.reshape(-1, 8, 1), z.reshape(-1, 8, 1)), axis = 2)

    return corners.astype(np.float32)


def boxes3d_to_corners3d_torch(boxes3d, flip = False):
    """
    :param boxes3d: (N, 7) [x, y, z, h, w, l, ry]
    :return: corners_rotated: (N, 8, 3)
    """
    boxes_num = boxes3d.shape[0]
    h, w, l, ry = boxes3d[:, 3:4], boxes3d[:, 4:5], boxes3d[:, 5:6], boxes3d[:, 6:7]
    if flip:
        ry = ry + np.pi
    centers = boxes3d[:, 0:3]
    zeros = torch.cuda.FloatTensor(boxes_num, 1).fill_(0)
    ones = torch.cuda.FloatTensor(boxes_num, 1).fill_(1)

    x_corners = torch.cat([l / 2., l / 2., -l / 2., -l / 2., l / 2., l / 2., -l / 2., -l / 2.], dim = 1)  # (N, 8)
    y_corners = torch.cat([zeros, zeros, zeros, zeros, -h, -h, -h, -h], dim = 1)  # (N, 8)
    z_corners = torch.cat([w / 2., -w / 2., -w / 2., w / 2., w / 2., -w / 2., -w / 2., w / 2.], dim = 1)  # (N, 8)
    corners = torch.cat((x_corners.unsqueeze(dim = 1), y_corners.unsqueeze(dim = 1), z_corners.unsqueeze(dim = 1)),
                        dim = 1)  # (N, 3, 8)

    cosa, sina = torch.cos(ry), torch.sin(ry)
    raw_1 = torch.cat([cosa, zeros, sina], dim = 1)
    raw_2 = torch.cat([zeros, ones, zeros], dim = 1)
    raw_3 = torch.cat([-sina, zeros, cosa], dim = 1)
    R = torch.cat((raw_1.unsqueeze(dim = 1), raw_2.unsqueeze(dim = 1), raw_3.unsqueeze(dim = 1)), dim = 1)  # (N, 3, 3)

    corners_rotated = torch.matmul(R, corners)  # (N, 3, 8)
    corners_rotated = corners_rotated + centers.unsqueeze(dim = 2).expand(-1, -1, 8)
    corners_rotated = corners_rotated.permute(0, 2, 1)
    return corners_rotated


def boxes3d_to_bev_torch(boxes3d):
    """
    :param boxes3d: (N, 7) [x, y, z, h, w, l, ry]
    :return:
        boxes_bev: (N, 5) [x1, y1, x2, y2, ry]
    """
    boxes_bev = boxes3d.new(torch.Size((boxes3d.shape[0], 5)))

    cu, cv = boxes3d[:, 0], boxes3d[:, 2]
    half_l, half_w = boxes3d[:, 5] / 2, boxes3d[:, 4] / 2
    boxes_bev[:, 0], boxes_bev[:, 1] = cu - half_l, cv - half_w
    boxes_bev[:, 2], boxes_bev[:, 3] = cu + half_l, cv + half_w
    boxes_bev[:, 4] = boxes3d[:, 6]
    return boxes_bev


def enlarge_box3d(boxes3d, extra_width):
    """
    :param boxes3d: (N, 7) [x, y, z, h, w, l, ry]
    """
    if isinstance(boxes3d, np.ndarray):
        large_boxes3d = boxes3d.copy()
    else:
        large_boxes3d = boxes3d.clone()
    large_boxes3d[:, 3:6] += extra_width * 2
    large_boxes3d[:, 1] += extra_width
    return large_boxes3d


def in_hull(p, hull):
    """
    :param p: (N, K) test points
    :param hull: (M, K) M corners of a box
    :return (N) bool
    """
    try:
        if not isinstance(hull, Delaunay):
            hull = Delaunay(hull)
        flag = hull.find_simplex(p) >= 0
    except scipy.spatial.qhull.QhullError:
        print('Warning: not a hull %s' % str(hull))
        flag = np.zeros(p.shape[0], dtype = np.bool)

    return flag


def objs_to_boxes3d(obj_list):
    boxes3d = np.zeros((obj_list.__len__(), 7), dtype = np.float32)
    for k, obj in enumerate(obj_list):
        boxes3d[k, 0:3], boxes3d[k, 3], boxes3d[k, 4], boxes3d[k, 5], boxes3d[k, 6] \
            = obj.pos, obj.h, obj.w, obj.l, obj.ry
    return boxes3d


def objs_to_scores(obj_list):
    scores = np.zeros((obj_list.__len__()), dtype = np.float32)
    for k, obj in enumerate(obj_list):
        scores[k] = obj.score
    return scores


def get_iou3d(corners3d, query_corners3d, need_bev = False):
    """	
    :param corners3d: (N, 8, 3) in rect coords	
    :param query_corners3d: (M, 8, 3)	
    :return: (N,M)
    """
    from shapely.geometry import Polygon
    A, B = corners3d, query_corners3d
    N, M = A.shape[0], B.shape[0]
    iou3d = np.zeros((N, M), dtype = np.float32)
    iou_bev = np.zeros((N, M), dtype = np.float32)

    # for height overlap, since y face down, use the negative y
    min_h_a = -A[:, 0:4, 1].sum(axis = 1) / 4.0
    max_h_a = -A[:, 4:8, 1].sum(axis = 1) / 4.0
    min_h_b = -B[:, 0:4, 1].sum(axis = 1) / 4.0
    max_h_b = -B[:, 4:8, 1].sum(axis = 1) / 4.0

    for i in range(N):
        for j in range(M):
            max_of_min = np.max([min_h_a[i], min_h_b[j]])
            min_of_max = np.min([max_h_a[i], max_h_b[j]])
            h_overlap = np.max([0, min_of_max - max_of_min])
            if h_overlap == 0:
                continue

            bottom_a, bottom_b = Polygon(A[i, 0:4, [0, 2]].T), Polygon(B[j, 0:4, [0, 2]].T)
            if bottom_a.is_valid and bottom_b.is_valid:
                # check is valid,  A valid Polygon may not possess any overlapping exterior or interior rings.
                bottom_overlap = bottom_a.intersection(bottom_b).area
            else:
                bottom_overlap = 0.
            overlap3d = bottom_overlap * h_overlap
            union3d = bottom_a.area * (max_h_a[i] - min_h_a[i]) + bottom_b.area * (max_h_b[j] - min_h_b[j]) - overlap3d
            iou3d[i][j] = overlap3d / union3d
            iou_bev[i][j] = bottom_overlap / (bottom_a.area + bottom_b.area - bottom_overlap)

    if need_bev:
        return iou3d, iou_bev

    return iou3d


def project_to_image(pts_3d, P):
    ''' Project 3d points to image plane.

    Usage: pts_2d = projectToImage(pts_3d, P)
      input: pts_3d: nx3 matrix
             P:      3x4 projection matrix
      output: pts_2d: nx2 matrix

      P(3x4) dot pts_3d_extended(4xn) = projected_pts_2d(3xn)
      => normalize projected_pts_2d(2xn)

      <=> pts_3d_extended(nx4) dot P'(4x3) = projected_pts_2d(nx3)
          => normalize projected_pts_2d(nx2)
    '''
    n = pts_3d.shape[0]
    pts_3d_extend = np.hstack((pts_3d, np.ones((n, 1))))
    print(('pts_3d_extend shape: ', pts_3d_extend.shape))
    pts_2d = np.dot(pts_3d_extend, np.transpose(P))  # nx3
    pts_2d[:, 0] /= pts_2d[:, 2]
    pts_2d[:, 1] /= pts_2d[:, 2]
    return pts_2d[:, 0:2]

def roty(t):
    ''' Rotation about the y-axis. '''
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, 0, s],
                     [0, 1, 0],
                     [-s, 0, c]])


def compute_box_3d(obj:object3d.Object3d, P):
    ''' Takes an object and a projection matrix (P) and projects the 3d
        bounding box into the image plane.
        Returns:
            corners_2d: (8,2) array in left image coord.
            corners_3d: (8,3) array in in rect camera coord.
    '''
    # compute rotational matrix around yaw axis
    R = roty(obj.ry)

    # 3d bounding box dimensions
    l = obj.l
    w = obj.w
    h = obj.h

    # 3d bounding box corners
    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
    z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]

    # rotate and translate 3d bounding box
    corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
    # print corners_3d.shape
    corners_3d[0, :] = corners_3d[0, :] + obj.pos[0]
    corners_3d[1, :] = corners_3d[1, :] + obj.pos[1]
    corners_3d[2, :] = corners_3d[2, :] + obj.pos[2]
    # print 'cornsers_3d: ', corners_3d
    # only draw 3d bounding box for objs in front of the camera
    # if np.any(corners_3d[2, :] < 0.1):
    #     corners_2d = np.zer
    #     return corners_2d, np.transpose(corners_3d)

    # project the 3d bounding box into the image plane
    corners_2d = project_to_image(np.transpose(corners_3d), P)
    # print 'corners_2d: ', corners_2d
    return corners_2d, np.transpose(corners_3d)

def draw_projected_box3d(image, qs, color=(0, 0, 255), thickness=2):
    ''' Draw 3d bounding box in image
        qs: (8,3) array of vertices for the 3d box in following order:
            1 -------- 0
           /|         /|
          2 -------- 3 .
          | |        | |
          . 5 -------- 4
          |/         |/
          6 -------- 7
    '''
    qs = qs.astype(np.int32)
    for k in range(0, 4):
        # Ref: http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html
        i, j = k, (k + 1) % 4
        # use LINE_AA for opencv3
        cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness)

        i, j = k + 4, (k + 1) % 4 + 4
        cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness)

        i, j = k, k + 4
        cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness)
    return image

def draw_text(image, box3d_pts_2d, score, color='red', thickness=1):
    """
    :param image:
    :param box3d_pts_2d:(8,2)
    :param score:
    :param color:
    :param thickness:
    :return:
    """
    text = str(score)
    left_top = np.min(box3d_pts_2d, axis=0)
    right_bottom = np.max(box3d_pts_2d, axis=0)
    # choose a reasonable location.
    location = left_top if (left_top[0] > 0 and left_top[1] > 0) else right_bottom
    location=[int(i) for i in location]
    cv2.putText(image, text, tuple(location),fontFace=cv2.FONT_ITALIC,fontScale=0.5, thickness=thickness,color=color)
    return image


def get_valid_flag(pts_img, pts_rect_depth, img_shape):
    """
    Valid point should be in the image (and in the PC_AREA_SCOPE)
    :param pts_rect:
    :param pts_img:
    :param pts_rect_depth:
    :param img_shape:
    :return:
    """
    val_flag_1 = np.logical_and(pts_img[:, 0] >= 0, pts_img[:, 0] < img_shape[1])
    val_flag_2 = np.logical_and(pts_img[:, 1] >= 0, pts_img[:, 1] < img_shape[0])
    val_flag_merge = np.logical_and(val_flag_1, val_flag_2)
    pts_valid_flag = np.logical_and(val_flag_merge, pts_rect_depth >= 0)

    return pts_valid_flag