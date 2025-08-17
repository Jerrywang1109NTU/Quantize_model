# -*- coding: utf-8 -*-
"""
Vitis AI PyTorch Quantization script for a custom YOLOv5 model.
This script is adapted for a modified Detect head that returns a list during
inference and a concatenated tensor during deployment.

Copyright 2023 Xilinx Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import os
import sys
import argparse
import time
import random
import yaml
import torch
from tqdm import tqdm
import numpy as np

# Vitis AI NNDCT API
from pytorch_nndct.apis import torch_quantizer

# Add current directory to path to import YOLOv5 modules
# 将当前目录添加到系统路径，以便导入YOLOv5的模块
dir_path = os.path.dirname(os.path.realpath(__file__))
if dir_path not in sys.path:
    sys.path.append(dir_path)

# YOLOv5 imports - FIXED: Changed scale_coords to scale_boxes
# YOLOv5 导入 - 已修复: 将 scale_coords 更改为 scale_boxes
from models.yolo import Model
from utils.dataloaders import create_dataloader
from utils.general import non_max_suppression, scale_boxes, check_yaml, box_iou
from utils.metrics import ap_per_class

# Set device: CUDA or CPU
# 设置运行设备：CUDA或CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Argument Parser ---
# --- 命令行参数解析 ---
def parse_args():
    parser = argparse.ArgumentParser(description='Custom YOLOv5 Quantization Script')
    
    # Data and Model paths based on user's structure
    # 根据用户的文件结构设置数据和模型路径参数
    parser.add_argument(
        '--data_yaml',
        default='data/x_ray.yaml',
        help='Path to your custom dataset YAML file (e.g., data/x_ray.yaml)')
    parser.add_argument(
        '--model_path',
        default='yolov5n_v6.pt', 
        help='Path to your trained YOLOv5 .pt model file (e.g., yolov5n_v6.pt)')
    parser.add_argument(
        '--model_config',
        default='models/yolov5n.yaml', 
        help='Path to the YOLOv5 model config YAML file. Ensure this matches your model.')

    # Quantization settings
    # 量化设置参数
    parser.add_argument(
        '--quant_mode', 
        default='test', 
        choices=['float', 'calib', 'test'], 
        help='Quantization mode: "float", "calib", or "test"')
    parser.add_argument(
        '--config_file',
        default=None,
        help='Vitis AI quantization configuration file')
    parser.add_argument(
        '--fast_finetune', 
        action='store_true',
        help='Enable fast finetuning during calibration')
    parser.add_argument(
        '--deploy', 
        action='store_true',
        help='Export xmodel for deployment (quant_mode must be "test")')
    parser.add_argument(
        '--inspect', 
        action='store_true',
        help='Inspect the float model with Vitis AI Inspector')

    # Dataloader and evaluation settings
    # 数据加载和评估参数
    parser.add_argument(
        '--subset_len',
        default=128,
        type=int,
        help='Number of images for calibration/evaluation. Default: use entire validation set.')
    parser.add_argument(
        '--batch_size',
        default=1,
        type=int,
        help='Batch size for calibration and evaluation')
    parser.add_argument(
        '--img_size',
        default=640,
        type=int,
        help='Image size for model input')
    
    args, _ = parser.parse_known_args()
    return args

# --- Helper Classes and Functions ---
# --- 辅助类与函数 ---

def load_data(data_yaml, batch_size, img_size, subset_len=None):
    """
    Creates a YOLOv5-compatible dataloader.
    MODIFIED to correctly resolve dataset paths from the YAML file.
    创建一个与YOLOv5兼容的数据加载器。
    已修改：用于从YAML文件正确解析数据集路径。
    """
    # Get the absolute path of the yaml file
    # 获取yaml文件的绝对路径
    data_yaml_path = os.path.abspath(data_yaml)
    data_yaml_dir = os.path.dirname(data_yaml_path)

    with open(data_yaml_path, 'r', errors='ignore') as f:
        data = yaml.safe_load(f)

    # Resolve the dataset root directory
    # The 'path' key in yaml is relative to the yaml file's directory
    # 解析数据集根目录
    # yaml中的 'path' 键是相对于yaml文件目录的相对路径
    dataset_root = data.get('path', '')
    if dataset_root and not os.path.isabs(dataset_root):
        dataset_root = os.path.join(data_yaml_dir, dataset_root)
    elif not dataset_root:
        # If 'path' key is missing, assume dataset root is the yaml's directory
        # 如果缺少 'path' 键，则假定数据集根目录就是yaml文件的目录
        dataset_root = data_yaml_dir

    # Resolve the validation image path
    # The 'val' key is relative to the dataset_root
    # 解析验证图像路径
    # 'val' 键是相对于数据集根目录的
    val_path_relative = data['val']
    val_path_absolute = os.path.join(dataset_root, val_path_relative)
    val_path_absolute = os.path.abspath(val_path_absolute) # clean up path (e.g., resolve '..')

    # Check if the resolved path exists and provide a helpful error if not
    # 检查解析后的路径是否存在，如果不存在则提供有用的错误信息
    if not os.path.exists(val_path_absolute):
         raise FileNotFoundError(f"Validation images path not found: '{val_path_absolute}'. "
                                 f"Please check the 'path' and 'val' keys in your '{data_yaml}' file.")
    
    # Now pass the absolute path to the dataloader
    # FIXED: Set workers=0 to prevent shared memory errors in Docker environments with low shm_size.
    # The ideal solution is to start the Docker container with a larger --shm-size flag.
    # 已修复：设置 workers=0 以防止在 shm_size 较低的 Docker 环境中出现共享内存错误。
    # 理想的解决方案是使用更大的 --shm-size 标志启动 Docker 容器。
    dataloader, dataset = create_dataloader(val_path_absolute, img_size, batch_size, 32, pad=0.5, rect=False, workers=0)
    
    if subset_len:
        # If subset_len is specified, create a new dataloader for the subset
        # 如果指定了subset_len，则为子集创建一个新的数据加载器
        assert subset_len <= len(dataset), f"subset_len ({subset_len}) is larger than the dataset size ({len(dataset)})"
        indices = random.sample(range(len(dataset)), subset_len)
        subset = torch.utils.data.Subset(dataset, indices)
        collate_fn = dataloader.collate_fn
        dataloader = torch.utils.data.DataLoader(subset, batch_size=batch_size, num_workers=0,
                                                 pin_memory=True, collate_fn=collate_fn)
    
    return dataloader, data

def create_yolo_grids(nx, ny, i, anchors, stride, device):
    """
    Creates the coordinate and anchor grids needed for YOLO post-processing.
    This replicates the logic from the original YOLOv5 Detect layer.
    创建YOLO后处理所需的坐标和锚点网格。
    此函数复制了原始YOLOv5 Detect层中的逻辑。
    """
    na = len(anchors[0])  # number of anchors
    yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)], indexing='ij')
    grid = torch.stack((xv, yv), 2).view(1, 1, ny, nx, 2).expand(1, na, ny, nx, 2)
    anchor_grid = (anchors[i].clone() * stride[i]).view(1, na, 1, 1, 2).expand(1, na, ny, nx, 2)
    return grid.float().to(device), anchor_grid.float().to(device)

def evaluate(model, val_loader, data_config, detect_layer, conf_thres=0.001, iou_thres=0.6):
    model.eval()
    model = model.to(device)
    if 'nc' not in data_config:
        raise KeyError("The 'nc' key is missing from your YAML file.")
    if 'names' not in data_config:
        raise KeyError("The 'names' key is missing from your YAML file.")
    names = data_config['names']
    nc = int(data_config['nc'])
    iouv = torch.linspace(0.5, 0.95, 10).to(device)
    niou = iouv.numel()

    stats = []
    pbar = tqdm(val_loader, desc='Evaluating mAP')
    nl, na, no = detect_layer.nl, detect_layer.na, detect_layer.no
    anchors, stride = detect_layer.anchors, detect_layer.stride

    for img, targets, path, shapes in pbar:
        img = img.to(device).float() / 255.0
        targets = targets.to(device)
        image_name = os.path.splitext(os.path.basename(path[0]))[0]
        with torch.no_grad():
            out = model(img)
            out = [tensor.permute(0, 2, 3, 1, 4).reshape(1, tensor.shape[2], tensor.shape[3], -1).cpu().numpy() for tensor in out]
            # 解包
            output0, output1, output2 = out

            # 打印形状确认
            from post_process import eval_sub_process

            labels_dir = "./data/data_calib/labels/valid/"

            print(labels_dir)
            map50, map50_95 = eval_sub_process(out, image_name, labels_dir, nc)
            print("mAP@.50: {:.4f}, mAP@.50:.95: {:.4f}".format(map50, map50_95))


# --- Main Quantization Function ---
# --- 主量化函数 ---
def quantization(args):
    """Main function for model quantization."""
    # Unpack arguments
    # 解析参数
    data_yaml, quant_mode, finetune, deploy = args.data_yaml, args.quant_mode, args.fast_finetune, args.deploy
    batch_size, subset_len, inspect, config_file = args.batch_size, args.subset_len, args.inspect, args.config_file
    model_path, model_config_path, img_size = args.model_path, args.model_config, args.img_size

    if deploy and quant_mode != 'test':
        print(r'Warning: Exporting xmodel must be in "test" mode. Disabling deploy.')
        deploy = False
    if deploy and batch_size != 1:
        print(r'Warning: Exporting xmodel requires batch size=1. Overriding batch_size to 1.')
        batch_size = 1
    
    # --- Load Float Model ---
    # --- 加载浮点模型 ---
    print(f"Loading float model from '{model_path}'...")
    try:
        ckpt = torch.load(model_path, map_location=device)
        model = Model(cfg=model_config_path, ch=3, nc=ckpt['model'].nc).to(device)
        state_dict = ckpt['model'].float().state_dict()
        model.load_state_dict(state_dict, strict=True)
        model.fuse()
        print("Model fused successfully.")
    except Exception as e:
        print(f"Error loading model: {e}\nCheck model/data paths and ensure '.pt' and '.yaml' files are compatible.")
        sys.exit(1)

    model.eval()
    
    # Get a reference to the Detect layer
    # 获取对Detect层的引用
    detect_head = model.model[-1]
    detect_head.eval()

    # Manually create the grids for the Detect layer.
    # 手动为Detect层创建网格。
    with torch.no_grad():
        dummy_input = torch.randn([1, 3, img_size, img_size], device=device)
        features = model(dummy_input)
        detect_head.grid = [torch.empty(0)] * detect_head.nl
        detect_head.anchor_grid = [torch.empty(0)] * detect_head.nl
        for i, f in enumerate(features):
            _, _, ny, nx, _ = f.shape 
            grid, anchor_grid = create_yolo_grids(nx, ny, i, detect_head.anchors, detect_head.stride, device)
            detect_head.grid[i] = grid
            detect_head.anchor_grid[i] = anchor_grid

    # --- Prepare for Quantization ---
    # --- 准备量化 ---
    input_tensor = torch.randn([batch_size, 3, img_size, img_size], device=device)

    if quant_mode == 'float':
        quant_model = model
    else:
        print(f"Creating quantizer for mode: {quant_mode}")
        quantizer = torch_quantizer(quant_mode, model, (input_tensor), device=device, quant_config_file=config_file)
        quant_model = quantizer.quant_model
    
    print("good\n")

    # --- Load Data for Calibration or Evaluation ---
    # --- 加载数据用于校准或评估 ---
    print(f"Loading data from '{data_yaml}'...")
    val_loader, data_config = load_data(data_yaml, batch_size, img_size, subset_len)
    
    # --- Execute Action based on quant_mode ---
    # --- 根据量化模式执行操作 ---

    if quant_mode == 'calib':
        print("Starting calibration...")
        for img, _, _, _ in tqdm(val_loader, desc="Calibrating"):
            quant_model(img.to(device).float() / 255.0)
        print("Calibration finished.")
        quantizer.export_quant_config()
        print("Quantization config exported.")
        if finetune:
            print("Starting fast finetune...")
            ft_loader, _ = load_data(data_yaml, batch_size=16, img_size=img_size, subset_len=1024)
            def ft_evaluate_wrapper(model):
                map50, _ = evaluate(model, ft_loader, data_config, detect_head)
                return map50
            quantizer.fast_finetune(ft_evaluate_wrapper, (quant_model,))
            print("Fast finetune finished.")

    elif quant_mode == 'test':
        if finetune and os.path.exists('quantize_result/fast_finetune_model_param.pth'):
            print("Loading fast finetune parameters...")
            quantizer.load_ft_param()
        
        if deploy:
            print("Setting 'export_xmodel=True' on the Detect head...")
            detect_module_found = False
            for submodule in quant_model.modules():
                if submodule.__class__.__name__ == 'Detect':
                    submodule.export_xmodel = True
                    detect_module_found = True
                    break
            if not detect_module_found:
                print("Warning: Could not find Detect module.")

            # FIXED: Always run a forward pass before exporting the model.
            # This satisfies the Vitis AI quantizer's requirement to trace the graph.
            # 已修复：在导出模型前始终运行一次前向传播。
            # 这满足了 Vitis AI 量化器追踪计算图的要求。
            print("Running a dummy forward pass for export...")
            dummy_batch, _, _, _ = next(iter(val_loader))
            quant_model(dummy_batch.to(device).float() / 255.0)
            
            print("Exporting xmodel...")
            quantizer.export_xmodel(deploy_check=False, output_dir='quant_output')
            print("Xmodel exported successfully.")
        else:
            print("Evaluating model performance...")
            evaluate(quant_model, val_loader, data_config, detect_head)
            print(f"\n--- Evaluation Results ({quant_mode} mode) ---")
            print(f"  mAP@.50    : {map50:.4f} %")
            print(f"  mAP@.50:.95: {map_50_95:.4f} %")
            print(f"-----------------------------------------\n")

    elif quant_mode == 'float' and inspect:
        print("Inspecting float model...")
        from pytorch_nndct.apis import Inspector
        inspector = Inspector("DPUCAHX8L_ISA0_SP")
        inspector.inspect(model, (input_tensor,), device=device)

# --- Main Execution Block ---
# --- 主执行块 ---
if __name__ == '__main__':
    args = parse_args()
    
    feature_test = ' float model evaluation'
    if args.quant_mode != 'float':
        feature_test = f' quantization (mode: {args.quant_mode})'
    title = f'YOLOv5n ({args.model_path})' + feature_test

    print(f"-------- Start: {title} --------")
    quantization(args)
    print(f"-------- End: {title} --------")
# python quant_test.py --quant_mode test --model_config models/yolov5s.yaml --model_path yolov5s_with_bg.pt
# python quant_test.py --quant_mode calib --model_config models/yolov5s.yaml --model_path yolov5s_with_bg.pt
# python quant_test.py --quant_mode test --subset_len 1 --batch_size 1 --deploy  --model_config models/yolov5s.yaml --model_path yolov5s_with_bg.pt

'''
vai_c_xir \
  -x ./quant_output/yolov5s_with_bg.xmodel \
  -a /opt/vitis_ai/compiler/arch/DPUCZDX8G/ZCU104/arch.json \
  -o ./compiled \ 
  -n yolov5s_with_bg

  vai_c_xir \
  -x ./quant_output/yolov5n_with_bg.xmodel \
  -a /opt/vitis_ai/compiler/arch/DPUCZDX8G/ZCU104/arch.json \
  -o ./compiled_with_bg_1600_n \
  -n yolov5n_with_bg
'''