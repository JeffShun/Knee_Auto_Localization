## 一、模型介绍
基于关键点检测膝盖自动定位模型

## 二、文件结构说明

```
Knee_Auto_Localization
├─ README.md
├─ setup.cfg
├─ test
│  ├─ analysis_tools
│  │  ├─ cal_metrics.py  # 计算测试集性能指标
│  │  └─ __init__.py 
│  ├─ data
│  │  └─ input  # 测试数据文件夹 
│  ├─ main.py  # 测试主函数
│  ├─ predictor.py  # 网络预测核心函数
│  └─ test_config.yaml  # 测试配置文件
└─ train
   ├─ config  # 模型配置文件
   ├─ custom
   │  ├─ dataset # 定义dataset类
   │  ├─ model
   │  │  ├─ backbones  # 关键点检测网络backbone
   │  │  ├─ model_head.py  # 关键点检测网络head
   │  │  ├─ model_loss.py  # 关键点检测网络损失
   │  │  ├─ model_network.py  # 关键点检测网络整体结构
   │  │  └─ __init__.py
   │  ├─ utils
   │  │  ├─ convert2pt.py  # 将pth文件静态化为pt
   │  │  ├─ convert2rt.py  # 将pth文件转化为onnx以及tensorrt
   │  │  ├─ dataloaderX.py  # dataloadX类，多卡并行时替换dataset
   │  │  ├─ data_transforms.py  # 数据预处理和数据增强
   │  │  ├─ distributed_utils.py  # 分布式处理函数
   │  │  ├─ generate_dataset.py  # 用于生成原始训练数据，输出npz文件
   │  │  ├─ logger.py  # 训练日志函数
   │  │  ├─ lr_scheduler.py  # 学习率调整函数
   │  │  ├─ model_backup.py  # 模型代码备份函数
   │  │  ├─ nrrd2nii.py  # 将nrrd文件转化为nii.gz文件
   │  │  ├─ tensorboad_utils.py  # tensorboard相关函数
   │  │  ├─ version_checkout.py  # 和model_backup.py配合使用，用于代码还原
   │  │  └─ __init__.py
   │  └─ __init__.py
   ├─ requirements.txt
   ├─ train.py  # 单卡训练入口
   ├─ train_data
   │  ├─ origin_data  # 原始训练数据
   │  │  ├─ train
   │  │  │  ├─ dcm_nii
   │  │  │  │  ├─ 5194.nii.gz
   │  │  │  └─ mask_nii
   │  │  │     ├─ 5194.mask.nii.gz
   │  │  └─ valid
   │  │     ├─ dcm_nii
   │  │     │  └─ 5272.nii.gz
   │  │     └─ mask_nii
   │  │        └─ 5272.mask.nii.gz
   ├─ train_dist.sh  # 多卡训练入口
   └─ train_multi_gpu.py
```


## 三、demo调用方法

step1. 准备训练原始数据
   * 在train文件夹下新建train_data/origin_data文件夹，放入训练的原始训练数据

step2. 生成标准格式后的npz训练数据，自动保存在train_data/processed_data文件夹下
   * cd train
   * python custom/utils/generate_dataset.py

step3. 开始训练
   * 单卡训练：python train.py
   * 分布式训练：sh ./train_dist.sh
   
step4. 准备测试数据
   * 将预测数据放入test/data/input目录

step5. 开始预测
   * cd test
   * python main.py

step6. 结果评估
   * python test/analysis_tools/cal_matrics.py
