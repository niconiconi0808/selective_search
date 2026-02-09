"""
Simple detection pipeline for Exercise 5.2 (balloon dataset).
Steps:
1) Load selective search proposals (npz with rects)
2) Build positive/negative samples by IoU thresholds
3) Extract HOG features
4) Train linear SVM and evaluate on valid split
"""

import os
import json
import argparse
import numpy as np
import skimage.io
import skimage.transform
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
import joblib


def load_coco_boxes(ann_path):
    with open(ann_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    img_id_to_file = {img["id"]: img["file_name"] for img in data["images"]}
    file_to_boxes = {fn: [] for fn in img_id_to_file.values()}
    for ann in data["annotations"]:
        img_id = ann["image_id"]
        fn = img_id_to_file[img_id]
        # COCO bbox: [x, y, w, h]
        file_to_boxes[fn].append(ann["bbox"])
    return file_to_boxes


def iou(box, gt):
    x, y, w, h = box
    gx, gy, gw, gh = gt
    x2, y2 = x + w, y + h
    gx2, gy2 = gx + gw, gy + gh

    ix1, iy1 = max(x, gx), max(y, gy)
    ix2, iy2 = min(x2, gx2), min(y2, gy2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    union = w * h + gw * gh - inter
    return inter / union


def extract_hog(image, rect, out_size):
    x, y, w, h = rect
    x, y, w, h = int(x), int(y), int(w), int(h)
    h_img, w_img = image.shape[:2]
    x2, y2 = min(x + w, w_img), min(y + h, h_img)
    x, y = max(0, x), max(0, y)
    if x2 <= x or y2 <= y:
        return None
    crop = image[y:y2, x:x2]
    if crop.size == 0:
        return None
    crop = skimage.transform.resize(crop, (out_size, out_size), anti_aliasing=True)
    feat = hog(
        crop,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        channel_axis=-1,
        feature_vector=True,
    )
    return feat


def build_samples(split_dir, proposals_dir, ann_path, tp, tn, out_size,
                  max_pos_per_img, max_neg_per_img):
    file_to_boxes = load_coco_boxes(ann_path)
    X, y = [], []

    for fn, gts in file_to_boxes.items():
        img_path = os.path.join(split_dir, fn)
        prop_path = os.path.join(proposals_dir, fn + ".npz")
        if not os.path.isfile(img_path) or not os.path.isfile(prop_path):
            continue

        image = skimage.io.imread(img_path)
        if image.ndim == 2:
            image = np.stack([image, image, image], axis=-1)
        if image.shape[2] > 3:
            image = image[:, :, :3]

        rects = np.load(prop_path)["rects"]
        pos_count = 0
        neg_count = 0

        for r in rects:
            max_iou = 0.0
            for gt in gts:
                max_iou = max(max_iou, iou(r, gt))

            if max_iou >= tp and pos_count < max_pos_per_img:
                feat = extract_hog(image, r, out_size)
                if feat is not None:
                    X.append(feat)
                    y.append(1)
                    pos_count += 1
            elif max_iou <= tn and neg_count < max_neg_per_img:
                feat = extract_hog(image, r, out_size)
                if feat is not None:
                    X.append(feat)
                    y.append(0)
                    neg_count += 1

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="../data/balloon_dataset")
    parser.add_argument("--proposals_root", default="../data/balloon_dataset/proposals")
    parser.add_argument("--tp", type=float, default=0.75)
    parser.add_argument("--tn", type=float, default=0.25)
    parser.add_argument("--out_size", type=int, default=128)
    parser.add_argument("--max_pos_per_img", type=int, default=50)
    parser.add_argument("--max_neg_per_img", type=int, default=200)
    parser.add_argument("--model_out", default="../results/balloon_svm.joblib")
    args = parser.parse_args()

    data_root = os.path.abspath(args.data_root)
    proposals_root = os.path.abspath(args.proposals_root)

    train_dir = os.path.join(data_root, "train")
    valid_dir = os.path.join(data_root, "valid")
    train_props = os.path.join(proposals_root, "train")
    valid_props = os.path.join(proposals_root, "valid")
    train_ann = os.path.join(train_dir, "_annotations.coco.json")
    valid_ann = os.path.join(valid_dir, "_annotations.coco.json")

    X_train, y_train = build_samples(
        train_dir, train_props, train_ann, args.tp, args.tn, args.out_size,
        args.max_pos_per_img, args.max_neg_per_img
    )
    X_valid, y_valid = build_samples(
        valid_dir, valid_props, valid_ann, args.tp, args.tn, args.out_size,
        args.max_pos_per_img, args.max_neg_per_img
    )

    print("Train samples:", X_train.shape, "Pos:", int(y_train.sum()), "Neg:", int((y_train == 0).sum()))
    print("Valid samples:", X_valid.shape, "Pos:", int(y_valid.sum()), "Neg:", int((y_valid == 0).sum()))

    clf = LinearSVC()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_valid)
    print(classification_report(y_valid, y_pred, digits=3))

    os.makedirs(os.path.dirname(os.path.abspath(args.model_out)), exist_ok=True)
    joblib.dump(clf, args.model_out)
    print("Saved model to:", os.path.abspath(args.model_out))


if __name__ == "__main__":
    main()
