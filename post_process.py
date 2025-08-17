import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
import os

def draw_detections(image, boxes, scores, class_ids, class_names=None, save_path='result.jpg'):
    """
    将检测结果绘制在图片上
    """
    if image is None:
        print(f"Failed to load image: {image_path}")
        return

    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i]
        score = scores[i]
        class_id = class_ids[i]

        # 转成 int
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        # 画矩形
        color = (0, 255, 0)  # Green
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        # 标签文字
        label = f"{class_names[class_id] if class_names else 'cls_'+str(class_id)} {score:.2f}"
        cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # 保存
    cv2.imwrite(save_path, image)
    print(f"Detection result saved to {save_path}")


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def decode_single_scale(output, stride, anchors, num_classes=2, conf_threshold=0.25):
    """
    output: (H, W, 3*(5+num_classes)) 或 (1, H, W, 3*(5+num_classes))
    anchors: [(aw,ah), (aw,ah), (aw,ah)]  # 像素单位，已按 640 尺度
    """
    if output.ndim == 4:
        output = output[0]  # 去 batch 维
    H, W, C = output.shape
    na = 3
    no = 5 + num_classes
    out = output.reshape(H, W, na, no)  # (H,W,3,5+nc)

    # 拆分
    tx = out[..., 0]; ty = out[..., 1]
    tw = out[..., 2]; th = out[..., 3]
    to = out[..., 4]
    tcls = out[..., 5:]  # (H,W,3,nc)

    # grid (cx,cy)
    gy = np.arange(H).reshape(H, 1, 1)
    gx = np.arange(W).reshape(1, W, 1)
    cx = np.broadcast_to(gx, (H, W, na))
    cy = np.broadcast_to(gy, (H, W, na))

    # anchors
    anchors = np.array(anchors, dtype=np.float32)  # (3,2)
    aw = anchors[:, 0].reshape(1, 1, na)
    ah = anchors[:, 1].reshape(1, 1, na)

    # YOLOv5 公式
    sx = (1.0 / (1.0 + np.exp(-tx))) * 2 - 0.5
    sy = (1.0 / (1.0 + np.exp(-ty))) * 2 - 0.5
    sw = ((1.0 / (1.0 + np.exp(-tw))) * 2) ** 2
    sh = ((1.0 / (1.0 + np.exp(-th))) * 2) ** 2

    bx = (sx + cx) * stride
    by = (sy + cy) * stride
    bw = sw * aw
    bh = sh * ah

    x1 = bx - bw / 2
    y1 = by - bh / 2
    x2 = bx + bw / 2
    y2 = by + bh / 2

    # 置信度 = obj * max(cls)
    obj = 1.0 / (1.0 + np.exp(-to))
    cls = 1.0 / (1.0 + np.exp(-tcls))
    cls_conf = cls.max(axis=-1)                    # (H,W,3)
    cls_id   = cls.argmax(axis=-1).astype(np.int32)
    scores   = obj * cls_conf

    mask = scores >= conf_threshold
    boxes = np.stack([x1[mask], y1[mask], x2[mask], y2[mask]], axis=1)
    scores = scores[mask]
    class_ids = cls_id[mask]
    return boxes, scores, class_ids

def multi_scale_post_process(dpu_outputs, conf_threshold=0.25, nms_threshold=0.45, img_size=640):
    """
    dpu_outputs: [out0, out1, out2]
    outi shape: (H,W,3*(5+nc)) 或 (1,H,W,3*(5+nc))
    """
    # stride & anchors 与训练/导出配置要一致（默认 yolov5n/s/m 系列 640 输入）
    strides  = [8, 16, 32]
    anchorss = [
        [(10,13), (16,30), (33,23)],     # stride 8
        [(30,61), (62,45), (59,119)],    # stride 16
        [(116,90), (156,198), (373,326)] # stride 32
    ]

    all_boxes, all_scores, all_cls = [], [], []
    for out, stride, anc in zip(dpu_outputs, strides, anchorss):
        b, s, c = decode_single_scale(out, stride, anc, num_classes=2, conf_threshold=conf_threshold)
        all_boxes.append(b); all_scores.append(s); all_cls.append(c)

    if len(all_boxes) == 0 or sum(len(b) for b in all_boxes) == 0:
        return np.zeros((0,4),dtype=np.float32), np.zeros((0,),dtype=np.float32), np.zeros((0,),dtype=np.int32)

    boxes = np.concatenate(all_boxes, axis=0)
    scores = np.concatenate(all_scores, axis=0)
    class_ids = np.concatenate(all_cls, axis=0)

    # 最后一关再做一次 NMS（建议 class-wise）
    boxes, scores, class_ids = nms_classwise(boxes, scores, class_ids, iou_thr=nms_threshold, agnostic=False)
    return boxes, scores, class_ids
def nms(boxes, scores, iou_threshold):
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return keep

def load_ground_truth_for_image(txt_path, img_width=640, img_height=640):
    """
    加载单张图片对应的 YOLO 格式 .txt 标签文件。
    返回: [[class_id, x1, y1, x2, y2], ...]
    """
    if not os.path.exists(txt_path):
        return np.array([])
    print(txt_path)
    boxes = []
    with open(txt_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5: continue
            class_id = int(parts[0])
            x_center_norm, y_center_norm, w_norm, h_norm = map(float, parts[1:])
            
            box_w = w_norm * img_width
            box_h = h_norm * img_height
            x1 = (x_center_norm * img_width) - (box_w / 2)
            y1 = (y_center_norm * img_height) - (box_h / 2)
            x2 = x1 + box_w
            y2 = y1 + box_h
            boxes.append([class_id, x1, y1, x2, y2])
            
    return np.array(boxes).reshape(-1, 5)

def cal_metrics(final_boxes, final_scores, final_class_ids, gt_boxes, num_classes):
    """
    Compute per-image AP for sanity-check (dataset-level AP 见 evaluate_npz_folder).
    gt_boxes: np.array of shape (M, 5) -> [class_id, x1, y1, x2, y2]
    """
    # --- helper: IoU between 1 pred box and many GT boxes (same class) ---
    def iou_1_to_many(box1_xyxy, gt_xyxy):
        xx1 = np.maximum(box1_xyxy[0], gt_xyxy[:, 0])
        yy1 = np.maximum(box1_xyxy[1], gt_xyxy[:, 1])
        xx2 = np.minimum(box1_xyxy[2], gt_xyxy[:, 2])
        yy2 = np.minimum(box1_xyxy[3], gt_xyxy[:, 3])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        area1 = (box1_xyxy[2] - box1_xyxy[0]) * (box1_xyxy[3] - box1_xyxy[1])
        area2 = (gt_xyxy[:, 2] - gt_xyxy[:, 0]) * (gt_xyxy[:, 3] - gt_xyxy[:, 1])
        return inter / (area1 + area2 - inter + 1e-6)

    def calculate_ap(recall, precision):
        m_prec = np.concatenate(([0.], precision, [0.]))
        m_rec = np.concatenate(([0.], recall, [1.]))
        for i in range(len(m_prec) - 2, -1, -1):
            m_prec[i] = np.maximum(m_prec[i], m_prec[i + 1])
        idx = np.where(m_rec[1:] != m_rec[:-1])[0] + 1
        return np.sum((m_rec[idx] - m_rec[idx - 1]) * m_prec[idx])

    # guard
    if final_boxes.size == 0:
        if gt_boxes.size == 0:
            return 100.0, 100.0
        else:
            return 0.0, 0.0

    # predictions matrix: [x1, y1, x2, y2, score, class_id]
    preds = np.column_stack([final_boxes, final_scores, final_class_ids])

    iou_thresholds = np.linspace(0.5, 0.95, 10)
    aps = np.zeros((num_classes, len(iou_thresholds)), dtype=np.float32)

    for c in range(num_classes):
        class_preds = predictions[predictions[:, 5] == c]
        class_gts = gt_boxes[gt_boxes[:, 0] == c]
        total_gt_boxes_for_class = len(class_gts)

        if total_gt_boxes_for_class == 0:
            aps[c, :] = 0.0 if len(class_preds) > 0 else 1.0
            continue
        if len(class_preds) == 0:
            aps[c, :] = 0.0
            continue

        class_preds = class_preds[class_preds[:, 4].argsort()[::-1]]
        # split GT for class c
        gtc = gt_boxes[gt_boxes[:, 0] == c]  # [:, 0] is class_id
        gtc_xyxy = gtc[:, 1:5] if gtc.size else np.zeros((0, 4), dtype=np.float32)
        n_gt = gtc_xyxy.shape[0]

        # split preds for class c
        pc = preds[preds[:, 5] == c]  # [:,5] is class_id
        if pc.size == 0:
            aps[c, :] = 0.0 if n_gt > 0 else 1.0
            continue

        # sort by score desc
        order = pc[:, 4].argsort()[::-1]
        pc = pc[order]

        for ti, thr in enumerate(iou_thresholds):
            tp = np.zeros(pc.shape[0], dtype=np.float32)
            fp = np.zeros(pc.shape[0], dtype=np.float32)
            matched = np.zeros(n_gt, dtype=np.int32)  # each GT can match at most once

            for i, p in enumerate(pc):
                if n_gt == 0:
                    fp[i] = 1.0
                    continue
                ious = iou_1_to_many(p[:4], gtc_xyxy)
                j = ious.argmax()
                if ious[j] >= thr and matched[j] == 0:
                    tp[i] = 1.0
                    matched[j] = 1
                else:
                    fp[i] = 1.0

            tp_cum = np.cumsum(tp)
            fp_cum = np.cumsum(fp)
            recall = tp_cum / (n_gt + 1e-6)
            precision = tp_cum / (tp_cum + fp_cum + 1e-6)
            aps[c, ti] = calculate_ap(recall, precision)

    map50 = float(np.mean(aps[:, 0]) * 100.0)
    map50_95 = float(np.mean(aps) * 100.0)
    return map50, map50_95


# ====================================================================
# 您的主评估函数 (重构后)
# ====================================================================
def eval_sub_process(tensor_outputs, image_name, labels_dir, num_classes):
    """
    接收单张图片的 DPU 输出，完成所有后处理和评估步骤。
    """
    
    # --- 1. 后处理，得到预测结果 ---
    print(f"\n--- 正在处理图片: {image_name} ---")
    start_time = time.time()
    
    for tensor in tensor_outputs:
        print(f"tensor shape: {tensor.shape}")  

    final_boxes, final_scores, final_class_ids = multi_scale_post_process(
        tensor_outputs,
        conf_threshold=0.5,
        nms_threshold=0.1,
        img_size=640
    )
    
    end_time = time.time()
    print(f"后处理耗时: {end_time - start_time:.4f} 秒")
    print(f"检测到 {len(final_boxes)} 个目标。")

    # --- 2. 加载对应的真值标签 ---
    val_label_path = os.path.join(labels_dir, f"{image_name}.txt")
    print(image_name)
    print(val_label_path)
    gt_boxes = load_ground_truth_for_image(val_label_path, img_width=640, img_height=640)
    print(f"找到 {len(gt_boxes)} 个真值标签。")

    # --- 3. 计算精度指标 ---
    if len(gt_boxes) == 0 and len(final_boxes) == 0:
        # 如果图片本身没有目标，也没检测出目标，则认为是完美的
        map50, map50_95 = 100.0, 100.0
    else:
        map50, map50_95 = cal_metrics(final_boxes, final_scores, final_class_ids, gt_boxes, num_classes)

    print(f"该图 mAP@.50: {map50:.2f}%, mAP@.50:.95: {map50_95:.2f}%")

    return map50, map50_95


if __name__ == "__main__":
    # 假设你已经跑完 multi_scale_post_process 得到以下变量
    start_time = time.time()
    data = np.load('dpu_outputs.npz')
    output0 = data['out0']
    output1 = data['out1']
    output2 = data['out2']

    final_boxes, final_scores, final_class_ids = multi_scale_post_process(
        [output0, output1, output2],
        conf_threshold=0.5,
        nms_threshold=0.1,
        img_size=640
    )
    end_time = time.time()
    print(f"Post-processing time: {end_time - start_time:.4f} seconds")
    # 原图路径（输入给DPU时用的原图路径）
    image_path = '8_13_b_0.png'
    # image_path = '8_mod.png'

    image = cv2.imread(image_path)
    image = cv2.resize(image, (640, 640))

    # 类别名（举例）
    class_names = ['A', 'B']
    
    # 绘图
    draw_detections(image, final_boxes, final_scores, final_class_ids, class_names, save_path='result_single_channel.png')