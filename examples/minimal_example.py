# The bare-bones example from the README.
# Run coco_example.py first to get mask_rcnn_bbox.json
from tidecv import TIDE, datasets

if __name__ == '__main__':
    tide = TIDE()
    KITTI_DataPath = ''
    pred_label_path = ''
    gt_data = datasets.KITTI(split='val', path=KITTI_DataPath, use_box2d=False)
    pred_data = datasets.KITTIResult(split='val', path=KITTI_DataPath, use_box2d=False,
                                     pred_label_path=pred_label_path)
    tide.evaluate(gt=gt_data, preds=pred_data, mode=TIDE.BOX3D,pos_threshold=0.7,background_threshold=0.1)

    tide.summarize()
    tide.plot()

