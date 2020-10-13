import numpy as np
import cv2
import pickle
from tidecv.kittidataset.kitti_dataset import KittiDataset
from tidecv.kittidataset.object3d import Object3d
from tidecv.kittidataset.calibration import Calibration
from tidecv.kittidataset import kitti_utils
import os

from matplotlib.lines import Line2D
from tidecv.errors.main_errors import *
import matplotlib.pyplot as plt
import math
import tqdm

r=(255,0,0)
g=(0,255,0)
b=(0,0,255)

def sigmoid(x):
    return 1. / (1. + math.exp(-x))


def format_str(x:float):
    return "%d" % int(x*1000)



def transform_to_img(xmin, xmax, ymin, ymax,
                     res=0.1,
                     side_range=(-20., 20 - 0.05),  # left-most to right-most
                     fwd_range=(0., 40. - 0.05),  # back-most to forward-most
                     ):
    xmin_img = -ymax / res - side_range[0] / res
    xmax_img = -ymin / res - side_range[0] / res
    ymin_img = -xmax / res + fwd_range[1] / res
    ymax_img = -xmin / res + fwd_range[1] / res

    return xmin_img, xmax_img, ymin_img, ymax_img


def project_obj2velo_bev(bbox, calib, ax, color='red', extra_text=None):
    corner3d_cam2 = bbox.corner3d
    corner3d_velo = calib.rect_to_velo(corner3d_cam2)
    draw_bev_box_from_corner3d_velo(corner3d_velo, ax=ax, color=color, extra_text=extra_text)


def point_cloud_2_top(points,
                      res=0.1,
                      zres=0.3,
                      side_range=(-20., 20 - 0.05),  # left-most to right-most
                      fwd_range=(0., 40. - 0.05),  # back-most to forward-most
                      height_range=(-2., 0.),  # bottom-most to upper-most
                      ):
    """ Creates an birds eye view representation of the point cloud data for MV3D.
    Args:
        points:     (numpy array)
                    N rows of points data
                    Each point should be specified by at least 3 elements x,y,z
        res:        (float)
                    Desired resolution in metres to use. Each output pixel will
                    represent an square region res x res in size.
        zres:        (float)
                    Desired resolution on Z-axis in metres to use.
        side_range: (tuple of two floats)
                    (-left, right) in metres
                    left and right limits of rectangle to look at.
        fwd_range:  (tuple of two floats)
                    (-behind, front) in metres
                    back and front limits of rectangle to look at.
        height_range: (tuple of two floats)
                    (min, max) heights (in metres) relative to the origin.
                    All height values will be clipped to this min and max value,
                    such that anything below min will be truncated to min, and
                    the same for values above max.
    Returns:
        numpy array encoding height features , density and intensity.
    """
    # EXTRACT THE POINTS FOR EACH AXIS
    x_points = points[:, 0]
    y_points = points[:, 1]
    z_points = points[:, 2]
    reflectance = points[:, 3]

    # INITIALIZE EMPTY ARRAY - of the dimensions we want
    x_max = int((side_range[1] - side_range[0]) / res)
    y_max = int((fwd_range[1] - fwd_range[0]) / res)
    z_max = int((height_range[1] - height_range[0]) / zres)
    top = np.zeros([y_max + 1, x_max + 1, z_max + 1], dtype=np.float32)

    # FILTER - To return only indices of points within desired cube
    # Three filters for: Front-to-back, side-to-side, and height ranges
    # Note left side is positive y axis in LIDAR coordinates
    f_filt = np.logical_and(
        (x_points > fwd_range[0]), (x_points < fwd_range[1]))
    s_filt = np.logical_and(
        (y_points > -side_range[1]), (y_points < -side_range[0]))
    filt = np.logical_and(f_filt, s_filt)

    for i, height in enumerate(np.arange(height_range[0], height_range[1], zres)):
        z_filt = np.logical_and((z_points >= height),
                                (z_points < height + zres))
        zfilter = np.logical_and(filt, z_filt)
        indices = np.argwhere(zfilter).flatten()

        # KEEPERS
        xi_points = x_points[indices]
        yi_points = y_points[indices]
        zi_points = z_points[indices]
        ref_i = reflectance[indices]

        # CONVERT TO PIXEL POSITION VALUES - Based on resolution
        x_img = (-yi_points / res).astype(np.int32)  # x axis is -y in LIDAR
        y_img = (-xi_points / res).astype(np.int32)  # y axis is -x in LIDAR

        # SHIFT PIXELS TO HAVE MINIMUM BE (0,0)
        # floor & ceil used to prevent anything being rounded to below 0 after
        # shift
        x_img -= int(np.floor(side_range[0] / res))
        y_img += int(np.floor(fwd_range[1] / res))

        # CLIP HEIGHT VALUES - to between min and max heights
        pixel_values = zi_points - height_range[0]
        # pixel_values = zi_points

        # FILL PIXEL VALUES IN IMAGE ARRAY
        top[y_img, x_img, i] = pixel_values

        # max_intensity = np.max(prs[idx])
        top[y_img, x_img, z_max] = ref_i

    top = (top / np.max(top) * 255).astype(np.uint8)
    return top


def draw_bev_box_from_corner3d_velo(corners_3d_velo, ax, color='red', extra_text=None):
    x1, x2, x3, x4 = corners_3d_velo[0:4, 0]
    y1, y2, y3, y4 = corners_3d_velo[0:4, 1]
    x1, x2, y1, y2 = transform_to_img(x1, x2, y1, y2, side_range=(-40., 40 - 0.05), fwd_range=(0., 80. - 0.05))
    x3, x4, y3, y4 = transform_to_img(x3, x4, y3, y4, side_range=(-40., 40 - 0.05), fwd_range=(0., 80. - 0.05))
    ps = []
    polygon = np.zeros([5, 2], dtype=np.float32)
    polygon[0, 0] = x1
    polygon[1, 0] = x2
    polygon[2, 0] = x3
    polygon[3, 0] = x4
    polygon[4, 0] = x1
    polygon[0, 1] = y1
    polygon[1, 1] = y2
    polygon[2, 1] = y3
    polygon[3, 1] = y4
    polygon[4, 1] = y1
    line1 = [(x1, y1), (x2, y2)]
    line2 = [(x2, y2), (x3, y3)]
    line3 = [(x3, y3), (x4, y4)]
    line4 = [(x4, y4), (x1, y1)]
    (line1_xs, line1_ys) = zip(*line1)
    (line2_xs, line2_ys) = zip(*line2)
    (line3_xs, line3_ys) = zip(*line3)
    (line4_xs, line4_ys) = zip(*line4)
    ax.add_line(Line2D(line1_xs, line1_ys, linewidth=1, color=color))
    ax.add_line(Line2D(line2_xs, line2_ys, linewidth=1, color=color))
    ax.add_line(Line2D(line3_xs, line3_ys, linewidth=1, color=color))
    ax.add_line(Line2D(line4_xs, line4_ys, linewidth=1, color=color))
    if extra_text is not None:
        ax.text(x1, y1, extra_text, color='yellow')


# def vis3d_err(err, kitti_dataset: KittiDataset, ax):
#     if isinstance(err, (ClassError, BoxError, AngleError)):
#         pred, gt = err.pred, err.gt
#         box = pred
#     elif isinstance(err, (DuplicateError, BackgroundError, OtherError)):
#         pred, gt = err.pred, None
#         box = pred
#     elif isinstance(err, (MissedError)):
#         pred, gt = None, err.gt
#         box = gt
#     else:
#         print("Unsupported error")
#         return
#
#     # Object3D
#     image_id = int(box['image'])
#     calib = kitti_dataset.get_calib(image_id)
#     if pred is not None:
#         bbox_pred = pred['bbox']
#         project_obj2velo_bev(bbox_pred, calib, ax=ax, color='red')
#     if gt is not None:
#         bbox_gt = gt['bbox']
#         project_obj2velo_bev(bbox_gt, calib, ax=ax, color='green')




def draw_bev_bboxes_(calib, obj_gt, obj_pred, points, save_dir,iou=None):
    fig, ax = plt.subplots(figsize=(20, 20))
    top = point_cloud_2_top(points, zres=1.0, side_range=(-40., 40 - 0.05), fwd_range=(0., 80. - 0.05))
    top = np.array(top, dtype=np.float32)
    ax.imshow(top, aspect='equal')
    # compute iou here for visulization.

    for i,obj in enumerate(obj_pred):
        if iou is not None:
            iou_score = iou[i]
            extra_text = format_str(sigmoid(obj.score)) +"/"+ format_str(iou_score)
        else:
            extra_text = format_str(sigmoid(obj.score))
        project_obj2velo_bev(obj, calib=calib, ax=ax, color='red', extra_text=extra_text)
    for obj in obj_gt:
        if obj.cls_type in ['Car']:
            project_obj2velo_bev(obj, calib, ax, color='green' if 1 <= obj.level <= 3 else 'blue')
    plt.axis('off')
    plt.tight_layout()
    plt.draw()
    plt.savefig(fname=save_dir, quality=50)


def show_image_with_boxes(img, obj_gt,obj_pred, calib, save_dir=''):
    ''' Show image with 2D bounding boxes '''

    img2 = np.copy(img)  # for 3d bbox
    for obj in obj_gt:
        if obj.cls_type == 'Car':
            img2 = draw_front_3dbox(calib, g if 1 <= obj.level <= 3 else b, img2, obj)
    for obj in obj_pred:
        if obj.cls_type == 'Car':
            img2 = draw_front_3dbox(calib, r, img2, obj)

    img = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)
    # cv2.imwrite(save_dir + '.png', img)
    cv2.imshow('name',img)


def draw_front_3dbox(calib, color:tuple, img2, obj):
    box3d_pts_2d, box3d_pts_3d = kitti_utils.compute_box_3d(obj, calib.P2)
    img2 = kitti_utils.draw_projected_box3d(img2, box3d_pts_2d, color, thickness=1)
    # draw score
    if obj.score is not None:
        kitti_utils.draw_text(img2, box3d_pts_2d, '%.3f' % obj.score, color=(255,255,0))
    return img2


def vis_bboxes(image_idx: int, kitti_dataset_gt: KittiDataset, kitti_dataset_pred: KittiDataset,
                    save_dir: str = None):
    """
    可视化俯视图以及前视图
    """
    points = kitti_dataset_gt.get_lidar(image_idx)
    calib = kitti_dataset_gt.get_calib(image_idx)
    obj_pred = kitti_dataset_pred.get_label(image_idx)
    obj_gt = kitti_dataset_gt.get_label(image_idx)
    img = kitti_dataset_gt.get_image(image_idx)

    detections = np.array([x.corner3d for x in obj_pred]).reshape((-1, 8, 3))
    gt_corners = np.array([x.corner3d for x in obj_gt]).reshape((-1, 8, 3))
    # (detection,gt)
    iou_d2g = kitti_utils.get_iou3d(detections, query_corners3d=gt_corners, need_bev=False)
    # (detections)
    max_iou = iou_d2g.max(axis=1)
    #俯视图
    draw_bev_bboxes_(calib, obj_gt, obj_pred, points, save_dir+'_bev.jpg',iou=max_iou)
    #前视图
    # show_image_with_boxes(img,obj_gt=obj_gt,obj_pred=obj_pred,calib=calib,save_dir=save_dir+'_front.jpg')




def get_box(err):
    if isinstance(err, (ClassError, BoxError, AngleError)):
        pred, gt = err.pred, err.gt
        box = pred
    elif isinstance(err, (DuplicateError, BackgroundError, OtherError)):
        pred, gt = err.pred, None
        box = pred
    elif isinstance(err, (MissedError)):
        pred, gt = None, err.gt
        box = gt
    else:
        print("Unsupported error")
        return None
    return box


if __name__ == '__main__':
    home_dir = '/home/xwchen/experiments/tide/vis/'
    # 如果不使用这些错误的样本的统计结果，可以去掉这部分，使用vis_bboxes函数
    with open(home_dir + "dump_error", 'rb') as f:
        error_dict = pickle.load(f)

    # count for error image idx
    error_img_dict = defaultdict(lambda: [0, []])
    for k, v in error_dict.items():
        # v list
        for err in v:
            box = get_box(err)
            if box is not None:
                image_idx = box['image']
            else:
                print('Unsupported_error')
                continue

            error_img_dict[image_idx][0] = error_img_dict[image_idx][0] + 1
            error_img_dict[image_idx][1].append(err)
    error_img_dict = dict(error_img_dict)
    with open(home_dir + 'error_img', 'wb+') as f:
        pickle.dump(error_img_dict, f)

    # vis error image
    kitti_dataset_gt = KittiDataset(root_dir='/data/xwchen', split='val')
    kitti_dataset_pred = KittiDataset(root_dir='/data/xwchen', split='val')
    kitti_dataset_pred.label_dir = pred_label_path = '/home/xwchen/experiments/EPNet_offical/tools/log/Car/full_epnet_wiou_branch/eval/eval_all_default/epoch_50/val/final_result/data/'
    count = 0
    os.makedirs(home_dir + 'img_dir', exist_ok=True)
    with tqdm.tqdm(total=len(error_img_dict)) as pbar:

        for k, v in error_img_dict.items():
            print('image_idx:%s, error:%d' % (k, v[0]))
            if v[0] > 0:

                err_types = v[1]
                err_name_count = defaultdict(lambda: 0)
                for e in err_types:
                    err_name_count[e.short_name] += 1
                s = str(dict(err_name_count))
                print(s)
                img_dir = home_dir + 'img_dir/' + "%s_e%d_%s" % (k, v[0], s)
                print('img_dir: ', img_dir)
                vis_bboxes(image_idx=k, kitti_dataset_gt=kitti_dataset_gt, kitti_dataset_pred=kitti_dataset_pred,
                                save_dir=img_dir)

            count += 1
            pbar.update(1)
