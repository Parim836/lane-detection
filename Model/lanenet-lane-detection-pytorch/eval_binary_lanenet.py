import os
import cv2
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms

from dataloader.data_loaders import TusimpleSet
from dataloader.transformers import Rescale
from model.lanenet.LaneNet import LaneNet
from model.utils.cli_helper_eval import parse_args

# --------------------------------------------------
# Device
# --------------------------------------------------
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# =====================================================
# Pixel-level Metrics
# =====================================================
def compute_iou_and_dice(pred, gt):
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()

    iou = intersection / (union + 1e-6)
    dice = (2.0 * intersection) / (pred.sum() + gt.sum() + 1e-6)
    return iou, dice


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


# --------------------------------------------------
# Lane Accuracy
# --------------------------------------------------
# def compute_lane_accuracy(pred_bin, gt_bin, overlap_thresh=0.3):
#     pred_labels, pred_n = cluster_lanes(pred_bin)
#     gt_labels, gt_n = cluster_lanes(gt_bin)

#     if gt_n == 0:
#         return 1.0 if pred_n == 0 else 0.0

#     correct = 0
#     for gt_id in range(1, gt_n + 1):
#         gt_mask = (gt_labels == gt_id)
#         best_iou = 0.0

#         for pred_id in range(1, pred_n + 1):
#             pred_mask = (pred_labels == pred_id)
#             inter = np.logical_and(gt_mask, pred_mask).sum()
#             union = np.logical_or(gt_mask, pred_mask).sum()
#             iou = inter / (union + 1e-6)
#             best_iou = max(best_iou, iou)

#         if best_iou >= overlap_thresh:
#             correct += 1

#     return correct / gt_n
def compute_lane_accuracy(pred_bin, gt_bin, overlap_thresh=0.3):
    pred_labels, pred_n = cluster_lanes(pred_bin)
    gt_labels, gt_n = cluster_lanes(gt_bin)

    if gt_n == 0 and pred_n == 0:
        return 1.0
    if gt_n == 0 and pred_n > 0:
        return 0.0

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

    acc = tp / (tp + fp + fn + 1e-6)
    return acc


# --------------------------------------------------
# Precision / Recall / F1 (Lane-level)
# --------------------------------------------------
def compute_lane_prf(pred_bin, gt_bin, overlap_thresh=0.3):
    pred_labels, pred_n = cluster_lanes(pred_bin)
    gt_labels, gt_n = cluster_lanes(gt_bin)

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

    fn = gt_n - tp
    fp = pred_n - len(matched_pred)

    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)

    return precision, recall, f1


# --------------------------------------------------
# Evaluation
# --------------------------------------------------
def evaluation():
    args = parse_args()

    resize_height = args.height
    resize_width = args.width

    image_transform = transforms.Compose([
        transforms.Resize((resize_height, resize_width)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    target_transform = transforms.Compose([
        Rescale((resize_width, resize_height)),
    ])

    dataset_file = os.path.join(args.dataset, "val.txt")
    eval_dataset = TusimpleSet(
        dataset_file,
        transform=image_transform,
        target_transform=target_transform
    )

    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=1,
        shuffle=False
    )

    model = LaneNet(arch=args.model_type)
    state_dict = torch.load(args.model, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    iou_sum = dice_sum = acc_sum = 0.0
    prec_sum = rec_sum = f1_sum = 0.0

    with torch.no_grad():
        for x, target, _ in eval_dataloader:
            x = x.to(DEVICE)

            output = model(x)

            pred = output["binary_seg_pred"]
            pred = torch.squeeze(pred).cpu().numpy().astype(np.uint8)

            gt = torch.squeeze(target).cpu().numpy()
            gt = (gt > 0).astype(np.uint8)

            iou, dice = compute_iou_and_dice(pred, gt)
            acc = compute_lane_accuracy(pred, gt)
            precision, recall, f1 = compute_lane_prf(pred, gt)

            iou_sum += iou
            dice_sum += dice
            acc_sum += acc
            prec_sum += precision
            rec_sum += recall
            f1_sum += f1

    n = len(eval_dataset)

    
    print("====== Evaluation (LaneNet_Pytorch) ======")
    print("Samples            :", n)
    print("Mean IoU (pixel)   :", iou_sum / n)
    print("Mean Dice (pixel)  :", dice_sum / n)
    print("Lane Accuracy      :", acc_sum / n)
    print("Lane Precision     :", prec_sum / n)
    print("Lane Recall        :", rec_sum / n)
    print("Lane F1-score      :", f1_sum / n)
    print("==========================================")


# --------------------------------------------------
# Main
# --------------------------------------------------
if __name__ == "__main__":
    evaluation()
