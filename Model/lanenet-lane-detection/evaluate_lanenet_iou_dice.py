#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluate LaneNet on custom dataset
Metrics:
- Pixel IoU
- Pixel Dice
- Lane Accuracy (clustering-based)
- Lane Precision
- Lane Recall
- Lane F1-score
"""

import argparse
import glob
import os.path as ops
import cv2
import numpy as np
import tensorflow as tf
import tqdm

from lanenet_model import lanenet
from local_utils.config_utils import parse_config_utils
from local_utils.log_util import init_logger

CFG = parse_config_utils.lanenet_cfg
LOG = init_logger.get_logger(log_file_name_prefix='lanenet_eval_metrics')


# =====================================================
# Pixel-level Metrics
# =====================================================
def compute_iou(pred, gt):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return 1.0 if union == 0 else inter / union


def compute_dice(pred, gt):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    total = pred.sum() + gt.sum()
    return 1.0 if total == 0 else 2.0 * inter / total


# =====================================================
# Clustering (Connected Components)
# =====================================================
def cluster_lanes(binary, min_area=250):
    """
    binary: (H, W) {0,1}
    return: labels, num_lanes
    """
    binary = (binary * 255).astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )

    filtered = np.zeros_like(binary)
    for i in range(1, num_labels):  # skip background
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            filtered[labels == i] = 255

    num_labels, labels = cv2.connectedComponents(filtered)
    return labels, num_labels - 1


# =====================================================
# Lane-level Metrics
# =====================================================
def compute_lane_metrics(pred_bin, gt_bin, overlap_thresh=0.3):
    pred_labels, pred_n = cluster_lanes(pred_bin)
    gt_labels, gt_n = cluster_lanes(gt_bin)

    # no lane in GT
    if gt_n == 0 and pred_n == 0:
        return 1.0, 1.0, 1.0, 1.0

    if gt_n == 0 and pred_n > 0:
        return 0.0, 0.0, 0.0, 0.0

    matched_pred = set()
    tp = 0

    for gt_id in range(1, gt_n + 1):
        gt_mask = (gt_labels == gt_id)
        best_iou = 0.0
        best_pred_id = -1

        for pred_id in range(1, pred_n + 1):
            pred_mask = (pred_labels == pred_id)
            inter = np.logical_and(gt_mask, pred_mask).sum()
            union = np.logical_or(gt_mask, pred_mask).sum()
            iou = inter / (union + 1e-6)

            if iou > best_iou:
                best_iou = iou
                best_pred_id = pred_id

        if best_iou >= overlap_thresh:
            tp += 1
            matched_pred.add(best_pred_id)

    fp = pred_n - len(matched_pred)
    fn = gt_n - tp

    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)

    # ✅ NEW Lane Accuracy (object-level)
    acc = tp / (tp + fp + fn + 1e-6)

    f1 = 2 * precision * recall / (precision + recall + 1e-6)

    return acc, precision, recall, f1



# =====================================================
# Args
# =====================================================
def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, required=True)
    parser.add_argument('--mask_dir', type=str, required=True)
    parser.add_argument('--weights_path', type=str, required=True)
    return parser.parse_args()


# =====================================================
# Evaluation
# =====================================================
def eval_lanenet(image_dir, mask_dir, weights_path):

    input_tensor = tf.placeholder(
        dtype=tf.float32,
        shape=[1, 256, 512, 3],
        name='input_tensor'
    )

    net = lanenet.LaneNet(phase='test', cfg=CFG)
    binary_seg_ret, _ = net.inference(input_tensor, name='LaneNet')

    saver = tf.train.Saver()
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = CFG.GPU.TF_ALLOW_GROWTH
    sess = tf.Session(config=sess_config)

    ious, dices = [], []
    accs, precs, recs, f1s = [], [], [], []

    with sess.as_default():
        saver.restore(sess, weights_path)
        LOG.info('Model restored from {}'.format(weights_path))

        image_list = sorted(
            glob.glob(ops.join(image_dir, '*.jpg')) +
            glob.glob(ops.join(image_dir, '*.png'))
        )

        for image_path in tqdm.tqdm(image_list):

            name = ops.splitext(ops.basename(image_path))[0]
            mask_path = ops.join(mask_dir, name + '.png')
            if not ops.exists(mask_path):
                continue

            image = cv2.imread(image_path)
            image = cv2.resize(image, (512, 256))
            image = image / 127.5 - 1.0

            gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            gt_mask = cv2.resize(gt_mask, (512, 256),
                                 interpolation=cv2.INTER_NEAREST)
            gt_mask = (gt_mask > 0).astype(np.uint8)

            binary_seg = sess.run(
                binary_seg_ret,
                feed_dict={input_tensor: [image]}
            )

            pred_mask = (binary_seg[0] > 0.5).astype(np.uint8)

            # Pixel-level
            ious.append(compute_iou(pred_mask, gt_mask))
            dices.append(compute_dice(pred_mask, gt_mask))

            # Lane-level (clustering)
            acc, p, r, f1 = compute_lane_metrics(pred_mask, gt_mask)
            accs.append(acc)
            precs.append(p)
            recs.append(r)
            f1s.append(f1)

    n = 200


    print()
    print("========== Evaluation (LaneNet) ==========")
    print("Samples            :", n)
    print("Mean IoU (pixel)   :", np.mean(ious))
    print("Mean Dice (pixel)  :", np.mean(dices))
    print("Lane Accuracy      :", np.mean(accs))
    print("Lane Precision     :", np.mean(precs))
    print("Lane Recall        :", np.mean(recs))
    print("Lane F1-score      :", np.mean(f1s))
    print("==========================================")


# =====================================================
# Main
# =====================================================
if __name__ == '__main__':
    args = init_args()
    eval_lanenet(args.image_dir, args.mask_dir, args.weights_path)
