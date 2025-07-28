import torch
import torch.nn as nn

# 假设这是你原来的 YOLO 模型
class YOLOModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 用实际 YOLO 模型替换这里
        # 举例：如果你用的是 YOLOv5n，就用实际 YOLOv5n 模型定义
        from models.yolo import Model  # 根据你YOLO代码路径修改
        self.model = Model('./models/yolov5n.yaml', ch=3, nc=2)  # 修改为你的 YOLO 配置文件路径、类别数等

    def forward(self, x):
        return self.model(x)  # 输出三个特征图 (out0, out1, out2)

# 新的无训练参数分类 head
class SimpleDefectClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid = nn.Sigmoid()
        self.pool0 = nn.AdaptiveMaxPool2d(1)
        self.pool1 = nn.AdaptiveMaxPool2d(1)
        self.pool2 = nn.AdaptiveMaxPool2d(1)

    def forward(self, out0, out1, out2):
        # 提取 objectness 通道 (index=4 假设是 obj_conf)
        obj0 = self.sigmoid(out0[..., 4])
        obj1 = self.sigmoid(out1[..., 4])
        obj2 = self.sigmoid(out2[..., 4])

        # 全局最大池化 over anchor, H, W
        p0 = torch.amax(obj0.view(obj0.shape[0], -1), dim=1, keepdim=True)  # (N,1)
        p1 = torch.amax(obj1.view(obj1.shape[0], -1), dim=1, keepdim=True)
        p2 = torch.amax(obj2.view(obj2.shape[0], -1), dim=1, keepdim=True)

        max_val = torch.max(torch.cat([p0, p1, p2], dim=1), dim=1)[0]  # 每batch一个float

        return max_val.unsqueeze(1)  # 输出 shape: (N, 1)

# 整合模型
class YOLOWithDefectClassifier(nn.Module):
    def __init__(self, yolo_model):
        super().__init__()
        self.yolo = yolo_model
        self.classifier = SimpleDefectClassifier()

    def forward(self, x):
        out0, out1, out2 = self.yolo(x)  # 原 YOLO 输出
        print(out0.shape)
        print(out1.shape)
        print(out2.shape)
        defect_score = self.classifier(out0, out1, out2)
        return out0, out1, out2, defect_score  # 多输出：3层特征+全局缺陷置信度

if __name__ == "__main__":
    # 路径配置：根据你的 pt 路径修改
    old_yolo_weight_path = 'yolov5n_v6.pt'
    new_concat_model_path = 'yolov5n_v6_classify.pt'

    # 实例化原 YOLO 模型
    base_yolo = YOLOModel()
    checkpoint = torch.load(old_yolo_weight_path, map_location='cpu')
    base_yolo.load_state_dict(checkpoint, strict=False)  # 不强制检测head对齐，避免报错

    # 拼接新模块
    full_model = YOLOWithDefectClassifier(base_yolo)

    # 测试一下前向通过（可选）
    dummy_input = torch.randn(1, 3, 640, 640)
    with torch.no_grad():
        out0, out1, out2, defect_score = full_model(dummy_input)
    print(f"Dummy run success. Defect score shape: {defect_score.shape}")

    # 保存新模型
    torch.save(full_model.state_dict(), new_concat_model_path)
    print(f"New concatenated model saved at: {new_concat_model_path}")
