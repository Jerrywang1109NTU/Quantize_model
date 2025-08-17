import os, time
import numpy as np
import cv2
from tqdm import tqdm
from post_process import (
    multi_scale_post_process,
    load_ground_truth_for_image,
    draw_detections,
    cal_metrics,
)


def match_image_stats_per_class(final_boxes, final_scores, final_class_ids, gt_boxes, num_classes, iou_thr=0.5):
    """
    Produce per-class TP/FP/FN for one image, using greedy matching at IoU>=iou_thr.
    Return:
        tp_c: (num_classes,) ints
        fp_c: (num_classes,) ints
        fn_c: (num_classes,) ints
        # For dataset-level AP accumulation across thresholds:
        pred_records: dict[c] -> list of (score, is_tp) *under current iou_thr*
                       (We will recompute for multiple thresholds outside by calling this with different iou_thr)
    """
    tp_c = np.zeros(num_classes, dtype=np.int64)
    fp_c = np.zeros(num_classes, dtype=np.int64)
    fn_c = np.zeros(num_classes, dtype=np.int64)
    pred_records = {c: [] for c in range(num_classes)}

    # quick exit
    if final_boxes.size == 0 and gt_boxes.size == 0:
        return tp_c, fp_c, fn_c, pred_records

    # per class process
    for c in range(num_classes):
        gtc = gt_boxes[gt_boxes[:, 0] == c]
        gtc_xyxy = gtc[:, 1:5] if gtc.size else np.zeros((0, 4), dtype=np.float32)
        n_gt = gtc_xyxy.shape[0]

        mask = (final_class_ids == c)
        pc_boxes = final_boxes[mask]
        pc_scores = final_scores[mask]
        if pc_boxes.size == 0:
            # no preds; all GT unmatched -> FN
            fn_c[c] += n_gt
            continue

        # sort preds by score desc
        order = pc_scores.argsort()[::-1]
        pc_boxes = pc_boxes[order]
        pc_scores = pc_scores[order]

        matched = np.zeros(n_gt, dtype=np.int32)

        for i in range(pc_boxes.shape[0]):
            p = pc_boxes[i]
            if n_gt == 0:
                pred_records[c].append((pc_scores[i], 0))
                fp_c[c] += 1
                continue

            xx1 = np.maximum(p[0], gtc_xyxy[:, 0])
            yy1 = np.maximum(p[1], gtc_xyxy[:, 1])
            xx2 = np.minimum(p[2], gtc_xyxy[:, 2])
            yy2 = np.minimum(p[3], gtc_xyxy[:, 3])
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            area_p = (p[2] - p[0]) * (p[3] - p[1])
            area_g = (gtc_xyxy[:, 2] - gtc_xyxy[:, 0]) * (gtc_xyxy[:, 3] - gtc_xyxy[:, 1])
            ious = inter / (area_p + area_g - inter + 1e-6)
            j = ious.argmax()

            if ious[j] >= iou_thr and matched[j] == 0:
                matched[j] = 1
                tp_c[c] += 1
                pred_records[c].append((pc_scores[i], 1))
            else:
                fp_c[c] += 1
                pred_records[c].append((pc_scores[i], 0))

        # any unmatched GT -> FN
        fn_c[c] += int((1 - matched).sum())

    return tp_c, fp_c, fn_c, pred_records


def compute_dataset_map(pred_records_per_thr, total_gt_per_class, num_classes):
    """
    pred_records_per_thr: dict[thr] -> dict[c] -> list of (score, is_tp)
    total_gt_per_class: (num_classes,) total GT boxes per class in the dataset
    Return:
        map50 (float %), map50_95 (float %)
    """
    def ap_from_records(records, n_gt):
        if n_gt == 0:
            # by convention for AP, if no GT and no preds -> AP=1; if preds exist -> AP=0
            return 1.0 if len(records) == 0 else 0.0
        # sort by score desc
        records.sort(key=lambda x: x[0], reverse=True)
        tps = np.array([r[1] for r in records], dtype=np.float32)
        fps = 1.0 - tps
        tp_cum = np.cumsum(tps)
        fp_cum = np.cumsum(fps)
        recall = tp_cum / (n_gt + 1e-6)
        precision = tp_cum / (tp_cum + fp_cum + 1e-6)
        # 11-point interpolation or integral? use integral-style (COCO-style)
        m_prec = np.concatenate(([0.], precision, [0.]))
        m_rec = np.concatenate(([0.], recall, [1.]))
        for i in range(len(m_prec) - 2, -1, -1):
            m_prec[i] = np.maximum(m_prec[i], m_prec[i + 1])
        idx = np.where(m_rec[1:] != m_rec[:-1])[0] + 1
        ap = float(np.sum((m_rec[idx] - m_rec[idx - 1]) * m_prec[idx]))
        return ap

    iou_thresholds = sorted(pred_records_per_thr.keys())  # e.g., [0.5, ..., 0.95]
    ap_all = []
    ap_50 = []

    for thr in iou_thresholds:
        per_thr = pred_records_per_thr[thr]
        ap_c = []
        for c in range(num_classes):
            ap_c.append(ap_from_records(per_thr[c], int(total_gt_per_class[c])))
        ap_c = np.array(ap_c, dtype=np.float32)
        ap_all.append(ap_c.mean())
        if abs(thr - 0.5) < 1e-6:
            ap_50 = ap_c.mean()

    map50 = float(ap_50 * 100.0)
    map50_95 = float(np.mean(ap_all) * 100.0)
    return map50, map50_95

def clip_boxes_to_img(boxes, img_w, img_h):
    if boxes.size == 0:
        return boxes
    boxes[:, 0] = np.clip(boxes[:, 0], 0, img_w - 1)
    boxes[:, 1] = np.clip(boxes[:, 1], 0, img_h - 1)
    boxes[:, 2] = np.clip(boxes[:, 2], 0, img_w - 1)
    boxes[:, 3] = np.clip(boxes[:, 3], 0, img_h - 1)
    return boxes


def evaluate_npz_folder(npz_dir, labels_dir, num_classes, conf_threshold=0.5, nms_threshold=0.45,
                        img_size=640, save_txt_path="eval_results.txt"):
    """
    Batch evaluate all .npz in a folder. Each npz must contain out0,out1,out2 arrays.
    Accumulate per-class TP/FP/FN at IoU=0.5, and compute dataset-level mAP@.5 and mAP@.5:.95.
    """
    npz_list = [f for f in os.listdir(npz_dir) if f.lower().endswith(".npz")]
    npz_list.sort()

    if len(npz_list) == 0:
        raise FileNotFoundError(f"No .npz found in: {npz_dir}")

    # accumulators
    total_tp = np.zeros(num_classes, dtype=np.int64)
    total_fp = np.zeros(num_classes, dtype=np.int64)
    total_fn = np.zeros(num_classes, dtype=np.int64)
    total_gt = np.zeros(num_classes, dtype=np.int64)

    # for dataset-level mAP at multiple IoUs
    iou_thresholds = np.linspace(0.5, 0.95, 10)
    pred_records_per_thr = {float(t): {c: [] for c in range(num_classes)} for t in iou_thresholds}

    # iterate files
    for fname in tqdm(npz_list, desc="Evaluating .npz folder"):
        fpath = os.path.join(npz_dir, fname)
        base = os.path.splitext(fname)[0]  # image name (should match label txt)
        data = np.load(fpath)

        # try common keys
        if not all(k in data for k in ("out0", "out1", "out2")):
            # allow other naming? adjust here if you use different names
            raise KeyError(f"{fname} missing keys out0/out1/out2")

        out0, out1, out2 = data["out0"], data["out1"], data["out2"]

        # postprocess -> predictions
        final_boxes, final_scores, final_class_ids = multi_scale_post_process(
            [out0, out1, out2],
            conf_threshold=conf_threshold,
            nms_threshold=nms_threshold,
            img_size=img_size
        )

        # load GT (expects labels_dir/base.txt in YOLO-normalized format)
        txt_path = os.path.join(labels_dir, f"{base}.txt")
        gt_boxes = load_ground_truth_for_image(txt_path, img_width=img_size, img_height=img_size)

        # accumulate total_gt
        if gt_boxes.size:
            for c in range(num_classes):
                total_gt[c] += int((gt_boxes[:, 0] == c).sum())

        # per-image TP/FP/FN for IoU=0.5 (for confusion-like stats)
        tp_c, fp_c, fn_c, _ = match_image_stats_per_class(
            final_boxes, final_scores, final_class_ids, gt_boxes, num_classes, iou_thr=0.5
        )
        total_tp += tp_c
        total_fp += fp_c
        total_fn += fn_c

        # records for all IoU thresholds (for dataset-level AP)
        for thr in iou_thresholds:
            _, _, _, rec = match_image_stats_per_class(
                final_boxes, final_scores, final_class_ids, gt_boxes, num_classes, iou_thr=float(thr)
            )
            # merge into global dict
            for c in range(num_classes):
                pred_records_per_thr[float(thr)][c].extend(rec[c])

    # compute dataset-level map
    map50, map50_95 = compute_dataset_map(pred_records_per_thr, total_gt, num_classes)

    # derive per-class metrics at IoU=0.5
    # Note: TN is not a well-defined metric in object detection (open background),
    # so we report TP/FP/FN and common derived metrics instead.
    eps = 1e-9
    precision_c = total_tp / (total_tp + total_fp + eps)
    recall_c = total_tp / (total_tp + total_fn + eps)
    f1_c = 2 * precision_c * recall_c / (precision_c + recall_c + eps)

    # overall
    overall_tp = int(total_tp.sum())
    overall_fp = int(total_fp.sum())
    overall_fn = int(total_fn.sum())
    overall_precision = float(overall_tp / (overall_tp + overall_fp + eps))
    overall_recall = float(overall_tp / (overall_tp + overall_fn + eps))
    overall_f1 = float(2 * overall_precision * overall_recall / (overall_precision + overall_recall + eps))

    # write results
    with open(save_txt_path, "w", encoding="utf-8") as f:
        f.write("=== Detection Evaluation (folder) ===\n")
        f.write(f"npz_dir          : {npz_dir}\n")
        f.write(f"labels_dir       : {labels_dir}\n")
        f.write(f"num_classes      : {num_classes}\n")
        f.write(f"conf_threshold   : {conf_threshold}\n")
        f.write(f"nms_threshold    : {nms_threshold}\n")
        f.write(f"img_size         : {img_size}\n")
        f.write("\n-- Dataset-level metrics --\n")
        f.write(f"mAP@0.50         : {map50:.2f}%\n")
        f.write(f"mAP@0.50:.95     : {map50_95:.2f}%\n")
        f.write(f"Overall TP/FP/FN : {overall_tp}/{overall_fp}/{overall_fn}\n")
        f.write(f"Overall P/R/F1   : {overall_precision:.4f}/{overall_recall:.4f}/{overall_f1:.4f}\n")

        f.write("\n-- Per-class (IoU=0.5 for PR/F1) --\n")
        f.write("class_id, TP, FP, FN, Precision, Recall, F1\n")
        for c in range(num_classes):
            f.write(f"{c}, {int(total_tp[c])}, {int(total_fp[c])}, {int(total_fn[c])}, "
                    f"{precision_c[c]:.4f}, {recall_c[c]:.4f}, {f1_c[c]:.4f}\n")
    print(f"[OK] Saved evaluation to: {save_txt_path}")

    # also return dict in case你想在外层脚本再用
    return {
        "map50": map50,
        "map50_95": map50_95,
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn,
        "precision_c": precision_c,
        "recall_c": recall_c,
        "f1_c": f1_c,
        "overall_precision": overall_precision,
        "overall_recall": overall_recall,
        "overall_f1": overall_f1
    }

def visualize_npz_folder(npz_dir, save_vis_dir, img_size=640,
                         conf_threshold=0.5, nms_threshold=0.45,
                         img_dir=None, class_names=None):
    """
    将 npz 的预测框画出来并保存为图片。
    - npz_dir: 存放 out0/out1/out2 的 .npz 文件夹
    - save_vis_dir: 输出图片文件夹（自动创建）
    - img_size: 网络输入尺寸（用于解码 & 画图尺寸）
    - img_dir: 原始图片所在目录；若为 None 或找不到同名图片，则用白底图
    - class_names: 类别名列表，如 ['A','B']；为空则用类ID
    """
    os.makedirs(save_vis_dir, exist_ok=True)
    exts = [".jpg", ".jpeg", ".png", ".bmp"]

    npz_list = [f for f in os.listdir(npz_dir) if f.lower().endswith(".npz")]
    npz_list.sort()
    if not npz_list:
        print(f"[WARN] No .npz found in: {npz_dir}")
        return

    for fname in tqdm(npz_list, desc="Visualizing detections"):
        fpath = os.path.join(npz_dir, fname)
        base = os.path.splitext(fname)[0]

        data = np.load(fpath)
        if not all(k in data for k in ("out0","out1","out2")):
            print(f"[SKIP] {fname} missing out0/out1/out2")
            continue
        out0, out1, out2 = data["out0"], data["out1"], data["out2"]

        # 解码
        final_boxes, final_scores, final_class_ids = multi_scale_post_process(
            [out0, out1, out2],
            conf_threshold=conf_threshold,
            nms_threshold=nms_threshold,
            img_size=img_size
        )

        # 裁到图像边界
        final_boxes = clip_boxes_to_img(final_boxes, img_size, img_size)

        # 找同名原图
        image = None
        if img_dir is not None:
            for ext in exts:
                ipath = os.path.join(img_dir, base + ext)
                if os.path.exists(ipath):
                    image = cv2.imread(ipath)
                    break

        # 准备画布
        if image is None:
            image = np.full((img_size, img_size, 3), 255, dtype=np.uint8)  # 白底
        else:
            image = cv2.resize(image, (img_size, img_size))

        # 类别名
        if class_names is None:
            # 用ID字符串占位
            uniq = np.unique(final_class_ids).tolist() if final_class_ids.size else []
            max_c = max(uniq) if uniq else 0
            class_names = [str(i) for i in range(max_c + 1)]

        try:
            draw_detections(image, final_boxes, final_scores, final_class_ids, class_names)
        except TypeError:
            # 如果你自己的 draw_detections 需要 save_path 参数
            pass

        save_path = os.path.join(save_vis_dir, base + ".png")
        cv2.imwrite(save_path, image)



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="single", choices=["single", "folder"], help="single or folder")
    parser.add_argument("--npz_path", default="dpu_outputs.npz", help="(single) path to one npz file")
    parser.add_argument("--npz_dir", default="npz_dir", help="(folder) path to folder containing npz files")
    parser.add_argument("--labels_dir", default="./data/YOLO_data_8_6_0_p/labels/test", help="labels directory")
    parser.add_argument("--img_path", default="8_13_b_0.png", help="image to draw (single)")
    parser.add_argument("--img_size", type=int, default=640)
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--conf_thres", type=float, default=0.5)
    parser.add_argument("--nms_thres", type=float, default=0.45)
    parser.add_argument("--save_txt", default="eval_results.txt")
    parser.add_argument("--save_vis_dir", default="", help="if set, save visualization images to this folder")
    parser.add_argument("--img_dir", default="", help="optional: original images folder (to draw on real images)")
    parser.add_argument("--class_names", default="A,B", help="optional: comma-separated class names, e.g. A,B")

    args = parser.parse_args()

    if args.mode == "single":
        # original single-image flow
        data = np.load(args.npz_path)
        output0 = data['out0']; output1 = data['out1']; output2 = data['out2']
        t0 = time.time()
        final_boxes, final_scores, final_class_ids = multi_scale_post_process(
            [output0, output1, output2],
            conf_threshold=args.conf_thres,
            nms_threshold=args.nms_thres,
            img_size=args.img_size
        )
        final_boxes = clip_boxes_to_img(final_boxes, args.img_size, args.img_size)
        print(f"Post-processing time: {time.time()-t0:.4f}s")

        image = cv2.imread(args.img_path)
        image = cv2.resize(image, (args.img_size, args.img_size))
        class_names = ['A', 'B']  # change to your names
        draw_detections(image, final_boxes, final_scores, final_class_ids, class_names, save_path='result_single.png')

        # one-image metrics (optional)
        base = os.path.splitext(os.path.basename(args.img_path))[0]
        gt = load_ground_truth_for_image(os.path.join(args.labels_dir, f"{base}.txt"),
                                         img_width=args.img_size, img_height=args.img_size)
        m50, m5095 = cal_metrics(final_boxes, final_scores, final_class_ids, gt, args.num_classes)
        print(f"Single mAP@.5: {m50:.2f}%, mAP@.5:.95: {m5095:.2f}%")

    else:
        # folder evaluation
        res = evaluate_npz_folder(
            npz_dir=args.npz_dir,
            labels_dir=args.labels_dir,
            num_classes=args.num_classes,
            conf_threshold=args.conf_thres,
            nms_threshold=args.nms_thres,
            img_size=args.img_size,
            save_txt_path=args.save_txt
        )

    if args.save_vis_dir:
        visualize_npz_folder(
            npz_dir=args.npz_dir,
            save_vis_dir=args.save_vis_dir,
            img_size=args.img_size,
            conf_threshold=args.conf_thres,
            nms_threshold=args.nms_thres,
            img_dir=(args.img_dir if args.img_dir else None),
            class_names=args.class_names
        )
        print(f"[OK] Saved visualizations to: {args.save_vis_dir}")

'''
python eval_dpu_output.py \
  --mode folder \
  --npz_dir ./dpu_outputs/dpu_outputs_yolov5n_no_bg_1600 \
  --labels_dir ./data/YOLO_data_8_6_0_p/labels/test \
  --num_classes 2 \
  --conf_thres 0.65 \
  --nms_thres 0.45 \
  --img_size 640 \
  --save_txt results_eval.txt

python eval_dpu_output.py \
  --mode folder \
  --npz_dir ./dpu_outputs/dpu_outputs_yolov5n_no_bg_1600 \
  --labels_dir ./data/YOLO_data_8_6_0_p/labels/test \
  --num_classes 2 \
  --conf_thres 0.65 \
  --nms_thres 0.45 \
  --img_size 640 \
  --save_txt results_eval.txt \
  --save_vis_dir ./vis_npz_yolov5n \
  --img_dir ./data/YOLO_data_8_6_0_p/images/test \
  --class_names A,B

'''