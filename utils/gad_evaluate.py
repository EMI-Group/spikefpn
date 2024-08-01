import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def coco_eval(gts, detections, height, width, labelmap=("car", "pedestrian")):
    """simple helper function wrapping around COCO's Python API
    :params:  gts iterable of numpy boxes for the ground truth
    :params:  detections iterable of numpy boxes for the detections
    :params:  height int
    :params:  width int
    :params:  labelmap iterable of class labels
    """
    categories = [
        {"id": id + 1, "name": class_name, "supercategory": "none"}
        for id, class_name in enumerate(labelmap)
    ]

    dataset, results = _to_coco_format(gts, detections, categories, height=height, width=width)

    coco_gt = COCO()
    coco_gt.dataset = dataset
    coco_gt.createIndex()
    print(len(results))
    coco_pred = coco_gt.loadRes(results)

    COCO_eval = COCOeval(coco_gt, coco_pred, "bbox")
    COCO_eval.params.imgIds = np.arange(1, len(gts) + 1, dtype=int)
    COCO_eval.evaluate()
    COCO_eval.accumulate()
    COCO_eval.summarize()

    ap50_95, ap50 = COCO_eval.stats[0], COCO_eval.stats[1]

    return ap50_95, ap50


def _to_coco_format(gts, detections, categories, height=240, width=304):
    """
    Utilitary function producing our data in a COCO usable format
    """
    annotations = []
    results = []
    images = []
    for image_id, (gt, pred) in enumerate(zip(gts, detections)):
        im_id = image_id + 1

        images.append({
            "date_captured": "2019",
            "file_name": "n.a",
            "id": im_id,
            "license": 1,
            "url": "",
            "height": height,
            "width": width
        })
        for label in gt:
            x1, y1, w, h = label[1:]
            class_id = label[0]
            area = w * h

            annotation = {
                "area": float(area),
                "iscrowd": False,
                "image_id": im_id,
                "bbox": [x1, y1, w, h],
                "category_id": int(class_id) + 1,
                "id": len(annotations) + 1
            }
            annotations.append(annotation)

        for label_dict in pred:
            bbox = label_dict["bbox"]
            category_id = label_dict["category_id"]
            score = label_dict["score"]
            image_result = {
                "image_id": im_id,
                "category_id": int(category_id) + 1,
                "score": float(score),
                "bbox": bbox,
            }
            results.append(image_result)

    dataset = {
        "info": {},
        "licenses": [],
        "type": "instances",
        "images": images,
        "annotations": annotations,
        "categories": categories
    }
    return dataset, results
