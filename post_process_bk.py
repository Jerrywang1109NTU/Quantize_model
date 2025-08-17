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

def decode_single_scale(output, stride, conf_threshold=0.5):
    """
    对单层特征图做解码
    output: numpy array, shape=(H, W, 21)
    stride: 当前层对应的 stride
    """
    H, W, C = output.shape
    
    num_anchors = 3
    num_classes = 2
    anchor_dim = 5 + num_classes

    output = output.reshape(H, W, num_anchors, anchor_dim)

    boxes = []
    scores = []
    class_ids = []

    for y in range(H):
        for x in range(W):
            for anchor in range(num_anchors):
                data = output[y, x, anchor]
                obj_conf = sigmoid(data[4])
                class_scores = sigmoid(data[5:])
                class_id = np.argmax(class_scores)
                class_conf = class_scores[class_id]
                final_conf = obj_conf * class_conf

                if final_conf < conf_threshold:
                    continue

                bx = (sigmoid(data[0]) + x) * stride
                by = (sigmoid(data[1]) + y) * stride
                bw = np.exp(data[2]) * stride
                bh = np.exp(data[3]) * stride

                x1 = bx - bw / 2
                y1 = by - bh / 2
                x2 = bx + bw / 2
                y2 = by + bh / 2

                boxes.append([x1, y1, x2, y2])
                scores.append(final_conf)
                class_ids.append(class_id)

    return boxes, scores, class_ids

def multi_scale_post_process(dpu_outputs, conf_threshold=0.5, nms_threshold=0.45, img_size=640):
    """
    dpu_outputs: [output0, output1, output2], 每层形状 (H, W, C)
    """
    strides = [8, 16, 32]  # 对应 80x80, 40x40, 20x20 层的 stride
    all_boxes = []
    all_scores = []
    all_class_ids = []

    for i, output in enumerate(dpu_outputs):
        boxes, scores, class_ids = decode_single_scale(output[0], strides[i], conf_threshold)
        all_boxes.extend(boxes)
        all_scores.extend(scores)
        all_class_ids.extend(class_ids)

    boxes = np.array(all_boxes)
    scores = np.array(all_scores)
    class_ids = np.array(all_class_ids)

    keep = nms(boxes, scores, nms_threshold)

    final_boxes = boxes[keep]
    final_scores = scores[keep]
    final_class_ids = class_ids[keep]

    return final_boxes, final_scores, final_class_ids

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
    为单张图片的预测和真值计算 mAP 指标。
    gt_boxes: [[class_id, x1, y1, x2, y2], ...]
    """
    print ("checkpoint_cal_metrics_1")
    # --- 辅助函数 ---
    def iou(box1, box2):
        xx1 = np.maximum(box1[0], box2[:, 1])
        yy1 = np.maximum(box1[1], box2[:, 2])
        xx2 = np.minimum(box1[2], box2[:, 3])
        yy2 = np.minimum(box1[3], box2[:, 4])
        w, h = np.maximum(0.0, xx2 - xx1), np.maximum(0.0, yy2 - yy1)
        inter = w * h
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[:, 3] - box2[:, 1]) * (box2[:, 4] - box2[:, 2])
        return inter / (area1 + area2 - inter + 1e-6)

    def calculate_ap(recall, precision):
        m_precision = np.concatenate(([0.], precision, [0.]))
        m_recall = np.concatenate(([0.], recall, [1.]))
        for i in range(len(m_precision) - 2, -1, -1):
            m_precision[i] = np.maximum(m_precision[i], m_precision[i + 1])
        indices = np.where(m_recall[1:] != m_recall[:-1])[0] + 1
        return np.sum((m_recall[indices] - m_recall[indices - 1]) * m_precision[indices])

    # --- mAP 计算主逻辑 ---
    iou_thresholds = np.linspace(0.5, 0.95, 10)
    aps = np.zeros((num_classes, len(iou_thresholds)))
    
    predictions = np.column_stack([final_boxes, final_scores, final_class_ids])

    print ("checkpoint_cal_metrics_2")
    print(gt_boxes.shape)

    for c in range(num_classes):
        c -= 1 # YOLO 类别从 0 开始
        class_preds = predictions[predictions[:, 5] == c]
        class_gts = gt_boxes[gt_boxes[:, 0] == c]
        total_gt_boxes_for_class = len(class_gts)
        
        if total_gt_boxes_for_class == 0:
            if len(class_preds) > 0:
                # 如果没有真值框但有预测框，所有预测都是 FP
                aps[c, :] = 0.0
            else:
                # 如果既没有真值框也没有预测框，AP 为 1 (完美)
                aps[c, :] = 1.0
            continue
            
        if len(class_preds) == 0:
            continue # 有真值但无预测，AP 为 0
        
        # 按置信度排序
        class_preds = class_preds[class_preds[:, 4].argsort()[::-1]]

        for ti, iou_thresh in enumerate(iou_thresholds):
            tp, fp = np.zeros(len(class_preds)), np.zeros(len(class_preds))
            gt_matched = np.zeros(total_gt_boxes_for_class)

            for i, pred in enumerate(class_preds):
                ious = iou(pred[:4], class_gts)
                best_iou_idx = np.argmax(ious)
                
                if ious[best_iou_idx] >= iou_thresh:
                    if gt_matched[best_iou_idx] == 0:
                        tp[i] = 1; gt_matched[best_iou_idx] = 1
                    else: fp[i] = 1
                else: fp[i] = 1
            
            tp_cumsum, fp_cumsum = np.cumsum(tp), np.cumsum(fp)
            recall = tp_cumsum / (total_gt_boxes_for_class + 1e-6)
            precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
            aps[c, ti] = calculate_ap(recall, precision)

    map50 = np.mean(aps[:, 0]) * 100
    map50_95 = np.mean(aps) * 100
    
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