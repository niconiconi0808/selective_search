"""
Inference script for Exercise 5.2.4.
Given an input image, generate proposals, classify with trained SVM,
and visualize detected balloons.
"""

import os
import argparse
import numpy as np
import skimage.io
import skimage.transform
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import joblib
from skimage.feature import hog

from selective_search import selective_search


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Input image path")
    parser.add_argument("--model", default="../results/balloon_svm.joblib")
    parser.add_argument("--out", default="../results/inference.png")
    parser.add_argument("--scale", type=float, default=500)
    parser.add_argument("--sigma", type=float, default=0.8)
    parser.add_argument("--min_size", type=int, default=20)
    parser.add_argument("--max_merges", type=int, default=None)
    parser.add_argument("--out_size", type=int, default=128)
    parser.add_argument("--score_thresh", type=float, default=0.0)
    args = parser.parse_args()

    image = skimage.io.imread(args.image)
    if image.ndim == 2:
        image = np.stack([image, image, image], axis=-1)
    if image.shape[2] > 3:
        image = image[:, :, :3]

    clf = joblib.load(args.model)

    _, regions = selective_search(
        image, scale=args.scale, sigma=args.sigma, min_size=args.min_size, max_merges=args.max_merges
    )
    rects = [r["rect"] for r in regions]

    feats = []
    rects_kept = []
    for r in rects:
        feat = extract_hog(image, r, args.out_size)
        if feat is None:
            continue
        feats.append(feat)
        rects_kept.append(r)

    if not feats:
        print("No valid proposals.")
        return

    X = np.array(feats, dtype=np.float32)
    scores = clf.decision_function(X)
    preds = (scores > args.score_thresh).astype(np.int32)

    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(8, 6))
    ax.imshow(image)
    count = 0
    for rect, pred, score in zip(rects_kept, preds, scores):
        if pred != 1:
            continue
        x, y, w, h = rect
        rect_patch = mpatches.Rectangle(
            (x, y), w, h, fill=False, edgecolor="red", linewidth=1
        )
        ax.add_patch(rect_patch)
        count += 1
    plt.axis("off")

    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print("Detections:", count, "Saved to:", out_path)


if __name__ == "__main__":
    main()
