import pynq_dpu
from yolo_post_process import multi_scale_post_process, draw_detections
from pynq_dpu import DpuOverlay
import numpy as np
import os, subprocess
import pytest

ol = DpuOverlay('dpu.bit')
ol.load_model("yolov5n_v6.xmodel")

import vart
import xir

# 读取 XModel

graph = xir.Graph.deserialize("yolov5n_v6.xmodel")
root_subgraph = graph.get_root_subgraph()

# 递归获取 DPU 子图
def get_child_subgraph_dpu(subgraph):
    if subgraph.has_attr("device") and subgraph.get_attr("device") == "DPU":
        return [subgraph]
    else:
        subgraphs = []
        for child in subgraph.get_children():
            subgraphs += get_child_subgraph_dpu(child)
        return subgraphs

subgraphs = get_child_subgraph_dpu(root_subgraph)

assert len(subgraphs) == 1, "Error: No DPU subgraph found!"

# 创建 runner
runner = vart.Runner.create_runner(subgraphs[0], "run")
print("Runner created successfully!")

import cv2
import time

start_time = time.time()
image = cv2.imread("./images/8_13_b_0.png")
image_1 = cv2.resize(image, (640, 640))
image = image_1 / 255.0
image = np.transpose(image, (2, 0, 1))  # HWC -> CHW
image = np.expand_dims(image, axis=0)   # Batch size 1
image = image.astype(np.float32)
end_time = time.time()
print(f"The Image Load took {end_time - start_time:.6f} seconds.")

import numpy as np

# 获取 DPU 输入输出 tensor 描述
start_time = time.time()
input_tensors = runner.get_input_tensors()
output_tensors = runner.get_output_tensors()

input_shape = input_tensors[0].dims  # Example: [1, 3, 640, 640]
output_shape = output_tensors[0].dims  # 通常是 [1, N, 85] 类似这种
# print("Input shape:", input_shape)
# print("Output shape:", output_shape)

# 分配 DPU 输入缓冲区
input_data = [np.empty(input_shape, dtype=np.float32, order='C')]

# 转换 NCHW -> NHWC
image_nhwc = np.transpose(image, (0, 2, 3, 1))
np.copyto(input_data[0], image_nhwc)

# 分配输出缓冲区
output_data = []
for out_tensor in output_tensors:
    out_shape = out_tensor.dims
    output_data.append(np.empty(out_shape, dtype=np.float32, order='C'))

# 异步执行
job_id = runner.execute_async(input_data, output_data)
runner.wait(job_id)
print("DPU inference done!")

# 异步执行
job_id = runner.execute_async(input_data, output_data)
runner.wait(job_id)
print("DPU Inference finished!")
end_time = time.time()
print(f"The DPU Runtime took {end_time - start_time:.6f} seconds.")

# 输出 DPU 结果
print("DPU raw output shape:", output_data[0].shape)
print("Example output (first 5 numbers):", output_data[0].flatten()[:5])
for idx, out in enumerate(output_data):
    print(f"Output {idx}: shape={out.shape}, min={np.min(out)}, max={np.max(out)}, mean={np.mean(out)}")
# print(output_data[0].flatten()[:100])
# np.savez('dpu_outputs.npz', out0=output_data[0], out1=output_data[1], out2=output_data[2])

start_time = time.time()
final_boxes, final_scores, final_class_ids = multi_scale_post_process(output_data)
class_names = ['A', 'B']
print(type(image))
print(image.shape if image is not None else "Image is None")
print(image.dtype)
    # 绘图
draw_detections(image_1, final_boxes, final_scores, final_class_ids, class_names, save_path='result.jpg')
end_time = time.time()
print(f"The NMS took {end_time - start_time:.6f} seconds.")