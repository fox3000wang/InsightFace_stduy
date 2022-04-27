# InsightFace 学习笔记

![](logo.jpeg)

## 简介

InsightFace 是一个人脸识别的工具箱，主要基于 PyTorch 和 MXNet

[官网](https://insightface.ai/)

[github](https://github.com/deepinsight/insightface)

## 依赖

- python3
- anaconda3
- pytorch

## 安装

```shell
pip install -U Cython cmake numpy

pip install -U insightface

pip install -U onnxruntime

# -U：升级。原来已经安装的包，带上U才会更新到最新版本，不带U不会装新版本。
```

## 运行

```python
import cv2  # 导入opencv库
import numpy as np  # 导入numpy库
import insightface  # 导入insightface库
from insightface.app import FaceAnalysis  # 导入FaceAnalysis类
from insightface.data import get_image as ins_get_image  # 导入get_image函数

# 创建FaceAnalysis类对象
app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))  # 准备模型
img = ins_get_image('t1')  # 获取图片
faces = app.get(img)  # 获取人脸
rimg = app.draw_on(img, faces)  # 在图片上画人脸
cv2.imwrite("./t1_output.jpg", rimg)  # 保存图片
```

## 输出

![](t1_output.jpg)

## 参考

https://github.com/deepinsight/insightface/tree/master/python-package
