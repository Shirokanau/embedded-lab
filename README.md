# 嵌入式课程项目作品集

课程作业的整理归档，两个项目：

| 项目 | 技术栈 | 目录 |
|---|---|---|
| 8051 单片机电子密码锁 | AT89C51、Keil C51、Proteus | `01-8051-electronic-lock/` |
| 传感器数据采集与分析 | Python、NumPy、Pandas、SciPy、Scikit-learn | `02-sensor-data-analysis/` |

这里放的是当时的源码、数据和结果图，没有重写或美化。数据处理部分哪些是实测、哪些是脚本里生成的仿真数据，在各项目的 README 里逐条写明；代码里发现的问题也没有改，统一记在各自的「已知问题」里。

## 仓库结构

```
embedded-lab/
├── 01-8051-electronic-lock/
│   ├── src/CX.c                       # 密码锁源码，426 行，含中文注释
│   ├── keil/cx.uvproj                 # Keil µVision 工程
│   ├── proteus/1.DSN, 1.PWI           # Proteus 仿真原理图与设置
│   └── firmware/electronic_lock.hex   # 编译产物，可直接烧录
└── 02-sensor-data-analysis/
    ├── src/preprocessing/             # 数据预处理与滤波，3 个脚本
    ├── src/machine_learning/          # 建模脚本，6 个
    ├── data/                          # 预处理与建模的输出数据
    ├── reports/                       # 脚本生成的 txt 报告
    ├── results/                       # 结果图
    └── requirements.txt
```

课程设计报告的正文（含原理图、流程图与封面信息）没有放进本仓库，只保留了报告里用到的源码、数据和结果图。

## 许可

代码以 MIT 协议开源；报告与数据仅供学习参考。
