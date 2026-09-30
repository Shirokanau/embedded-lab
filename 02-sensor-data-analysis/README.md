# 传感器数据采集 · 数据预处理 · 机器学习建模

> 课程实验（传感器实验）· Python · NumPy / Pandas / SciPy / Matplotlib / Scikit-learn
> 三类传感器（DHT11 温湿度、HC-SR04 超声波、MPU-6050 六轴）的串口数据采集、异常检测与滤波降噪、姿态解算与回归建模。

---

## ⚠️ 数据来源说明（**请先读这一节**）

仓库内脚本的数据来源并不一致，**哪些是实测、哪些是仿真，逐条列明如下**（依据代码本身，非推测）：

| 脚本 | 输入数据来源 | 说明 |
|---|---|---|
| `src/preprocessing/hcsr04_preprocessing_demo.py` | **真实采集** | 读 Excel 的 `时间`、`测量距离(mm)` 两列（原实验机路径 `D:\高校专项\实验项目\HC-SR04数据.xlsx`） |
| `src/preprocessing/dht11_preprocessing_demo.py` | **真实采集** | 读 Excel 的 `时间`、`温度`、`湿度` 三列（原路径 `D:\高校专项\实验项目\第2组温湿度校验数据-1010.xlsx`） |
| `src/preprocessing/mpu6050_preprocessing_demo.py` | **仿真数据** | 脚本内生成（`np.random.seed(46)`，600 点 @20Hz，Figure-8 轨迹），**不读取任何外部数据** |
| `src/machine_learning/hcsr04_machine_learning.py` | 真实优先 / 仿真回退 | 优先读 `HC-SR04预处理结果数据.xlsx`，文件不存在则走脚本内仿真分支 |
| `src/machine_learning/mpu6050_machine_learning.py` | 真实优先 / 仿真回退 | 同上，目标文件为 `MPU-6050预处理结果数据.xlsx` |
| `src/machine_learning/dht11_machine_learning.py` | **仿真数据** | 脚本内生成（`seed=42`，300 条） |
| `src/machine_learning/dht11_ml_simple.py` | **仿真数据** | 脚本内生成（`seed=42`） |
| `src/machine_learning/hcsr04_ml_simple.py` | **仅读 Excel** | 无仿真回退，缺文件会直接 `FileNotFoundError` |
| `src/machine_learning/mpu6050_ml_simple.py` | **仅读 Excel** | 同上 |

> **因此**：`reports/MPU-6050预处理报告.txt` 中的全部指标（含互补滤波对比）是**仿真数据**的运行结果；`reports/` 中 DHT11 与 HC-SR04 的报告来自真实采集数据。
> **仓库内不含原始采集 Excel**（原文件位于实验机 `D:\高校专项\实验项目\`，未随课程归档保留），故部分脚本无法在仓库内原样复现——详见文末「复现条件」。

---

## 1. 实验一：多传感器串口数据采集

| 项 | 内容 |
|---|---|
| 传感器 | DHT11 温湿度、HC-SR04 超声波、MPU-6050 六轴姿态 |
| 串口参数 | 温湿度 / 姿态 **115200**，超声波 **38400**；统一 **8 数据位 / 1 停止位 / 无校验（8N1）** |
| 采集结果 | 湿度 57.0%~95.0% RH，温度 25.9~26.6 ℃；测距 0~3926 mm；姿态 Roll / Pitch / Yaw |
| 结论 | 串口通信稳定、无乱码，验证了**波特率匹配**对数据完整性的影响 |

## 2. 实验二：数据预处理与滤波降噪

### 2.1 处理流程（三个脚本共同结构）

```
读取 Excel 实测数据
  → 异常值检测（Z-Score / IQR / 3-Sigma）
  → 置 NaN 后插值修复（线性插值）
  → 滤波降噪（移动平均 / 指数平滑 / Savitzky-Golay / 卡尔曼）
  → 误差量化（RMSE / MAE / MAPE）
  → 9 子图可视化 + 导出 xlsx + 生成 txt 报告
```

### 2.2 算法与参数

| 环节 | 方法 | 参数 |
|---|---|---|
| 异常检测 | Z-Score | 阈值 **2.5** |
| | IQR | 四分位距系数 **1.5** |
| | 3-Sigma | 阈值 **3** |
| 插值修复 | `Series.interpolate(linear, limit_direction='both')` / `np.interp` | — |
| 滤波 | 移动平均 | 窗口 **5**（`np.convolve(..., mode='same')`） |
| | 指数平滑 | **α = 0.3** |
| | Savitzky-Golay | 窗口 **7**、多项式阶 **3** |
| | **卡尔曼滤波**（自写） | 过程噪声 **Q=0.01**、测量噪声 **R=1.0**、初始协方差 **P=1.0** |
| 姿态解算（MPU-6050，仿真数据） | 陀螺仪零偏校准 | 取前 **100** 个样本的均值作为零偏估计 |
| | Butterworth 滤波 | 加速度计低通 **5 Hz**、陀螺仪高通 **0.5 Hz**，`fs=20Hz`、`order=2`、`filtfilt` |
| | **互补滤波** | **α = 0.96** |
| 标准化 | Z-Score / Min-Max / Robust | 预定义三种，实际调用 Z-Score |

### 2.3 结果

**HC-SR04（真实采集，490 组）** — `reports/HC-SR04预处理报告.txt`

| 处理方式 | RMSE (cm) | MAE (cm) | MAPE (%) |
|---|---|---|---|
| 原始数据 | 3.6574 | 1.5143 | 1.5154 |
| 清洗后 | 1.4709 | 1.2779 | 1.2788 |
| 移动平均 | 2.8595 | 0.7552 | 0.7558 |
| 指数平滑 | 0.5997 | 0.4843 | 0.4846 |
| **卡尔曼滤波** | **0.3076** | **0.2377** | **0.2379** |

> **必须说明的指标口径问题**：脚本中的"真实值"取自 **清洗后数据的均值**（`np.mean(cleaned)`，见 `hcsr04_preprocessing_demo.py` L207），**不是独立测量的真值**。滤波后的信号更接近自身均值，因此该 RMSE 天然会低于原始信号——**这个"降低 91.6%"不能等同于测距精度提升**，只能说明滤波平滑效果。这是本实验在指标定义上的主要局限。

**DHT11（真实采集，816 组）** — `reports/DHT11预处理报告.txt`

- 温度 18.3~21.7 ℃（均值 20.01 ℃，标准差 0.88 ℃）；湿度 35.1%~44.9%（均值 39.99%，标准差 2.41%）
- Z-Score 检测：温度、湿度异常值均为 **0 个**（该段数据较平稳）
- 滤波后标准差：温度 SG 滤波 0.5053 ℃；湿度指数平滑 1.0024%
- 温湿度相关系数 **0.0405**（基本不相关）

**MPU-6050（⚠️ 仿真数据，600 组 @20Hz）** — `reports/MPU-6050预处理报告.txt`

| 解算方式 | 俯仰角 RMSE (°) | 横滚角 RMSE (°) |
|---|---|---|
| 纯加速度计 | 0.722 | 0.773 |
| 纯陀螺仪 | 11.175 | 7.062 |
| 互补滤波（α=0.96） | 5.211 | 4.873 |

> **报告里如实打印出的是负面结果**：互补滤波误差**大于**纯加速度计（脚本输出"性能提升 −621.8% / −530.6%"）。这不是笔误，而是该脚本设计的必然结果：
> 1. 零偏校准假设"前 100 个样本为静止"，但这段数据是 Figure-8 **动态运动**，导致零偏估计严重偏离（估计 X=2.914°/s vs 实际 0.300°/s）；
> 2. 仿真数据中加速度计几乎没有噪声，静态解算本就占优；
> 3. 互补滤波仅使用 gx/gy，未使用 gz（yaw）。
>
> 保留这个负面结果是有意为之：**它比"看起来漂亮"的指标更能说明误差来源分析的过程**。

## 3. 实验三：机器学习建模与性能评估

### 3.1 特征工程（各脚本共同思路）

- 时间特征：`hour` / `minute` / `sin-cos` 周期编码
- 滚动统计：滚动均值、滚动标准差（窗口 5 / 10 / 20）与 EMA
- 滞后特征：1 / 3 / 5 阶滞后
- 物理与交互特征：加速度模、角速度模、姿态估计、温湿度交互与比值等
- 特征选择：**SelectKBest**（`mutual_info_regression` 或 `f_regression`，**k=15**）
- 标准化：`StandardScaler`；缺失值：`SimpleImputer(mean)` / 线性插值

### 3.2 模型与评估

| 项 | 内容 |
|---|---|
| 模型集合 | LinearRegression、**Ridge**、**Lasso**、**ElasticNet**、**RandomForest(100)**、**GradientBoosting(100)**、**SVR(rbf)**、**MLP((100,50,25))** |
| 超参搜索 | `GridSearchCV`（随机森林，`cv=5`，`scoring='r2'`） |
| 交叉验证 | **5 折**（`cross_val_score` / `GridSearchCV`） |
| 评估指标 | **R² / RMSE / MAE** |
| 模型融合 | Top-3 **简单平均**与**加权平均**（权重按测试集 R² 归一化）、`VotingRegressor` |
| 产出 | 模型性能对比表、预测结果、9 宫格可视化（学习/预测/误差分布等） |

> ⚠️ **DHT11 的两个机器学习脚本使用的是内置仿真数据**，`hcsr04_ml_simple.py` 与 `mpu6050_ml_simple.py` 则要求先有预处理输出的 Excel（仓库内不含）。因此实验三的结果表**不代表真实传感器数据的建模精度**，仅用于演示完整建模流程（特征工程 → 特征筛选 → 多模型对比 → 交叉验证 → 融合）。

## 4. 目录结构

```
02-sensor-data-analysis/
├── src/
│   ├── preprocessing/            # 实验二：预处理与滤波
│   │   ├── dht11_preprocessing_demo.py
│   │   ├── hcsr04_preprocessing_demo.py
│   │   └── mpu6050_preprocessing_demo.py
│   └── machine_learning/         # 实验三：建模（完整版 + 简化版）
│       ├── hcsr04_machine_learning.py / hcsr04_ml_simple.py
│       ├── dht11_machine_learning.py  / dht11_ml_simple.py
│       └── mpu6050_machine_learning.py / mpu6050_ml_simple.py
├── data/
│   ├── preprocessing/            # 预处理输出数据（含原始测量列，可查看真实数值）
│   └── machine_learning/         # 建模输出（模型性能对比 / 预测结果 / 预处理结果）
├── reports/                      # 三个传感器各一份 txt 报告（脚本自动生成）
└── results/                      # 结果图（预处理 3 张 + 机器学习 6 张）
```

## 5. 复现条件

**依赖**（见 `requirements.txt`）：`numpy`、`pandas`、`matplotlib`、`scipy`、`scikit-learn`、`openpyxl`

**字体**：脚本设置 `font.sans-serif = ['SimHei', ...]`，非 Windows 环境需自备中文字体，否则图注显示为方框。

**数据**：以下 4 个 Excel 是脚本原本读取的文件，**仓库内不包含**（原文件保留在实验机，未随归档一同留存）：

| 文件 | 需要的列 |
|---|---|
| `HC-SR04数据.xlsx` | `时间`、`测量距离(mm)` |
| `第2组温湿度校验数据-1010.xlsx` | `时间`、`温度`、`湿度` |
| `HC-SR04预处理结果数据.xlsx` | `时间`、`距离`、`速度`、`加速度`、`真实距离` |
| `MPU-6050预处理结果数据.xlsx` | `时间`、`Ax..Gz`、`真实俯仰角`、`真实横滚角` |

- 脚本中的路径是**写死的绝对路径**（原实验机 `D:\高校专项\实验项目\...`），复现时需改成本机路径；
- 无外部数据仍可运行的脚本：`mpu6050_preprocessing_demo.py`（纯仿真）、`dht11_machine_learning.py`、`hcsr04_machine_learning.py` 与 `mpu6050_machine_learning.py`（走仿真回退）；
- `data/` 下已提供**预处理输出的 Excel**（其中 `原始测量` / `原始温度` / `原始湿度` 等列保留了真实采集值），可在不改脚本的情况下查看真实数据与滤波结果的对照。

## 6. 已知问题（整理时发现，代码保持原样）

| # | 问题 | 位置 |
|---|---|---|
| 1 | 路径写死为实验机绝对路径（共 10 处），换机器即失效 | `dht11_preprocessing` L28/L408、`hcsr04_preprocessing` L27/L415、`mpu6050_preprocessing` L391、`hcsr04_ml` L31/L37、`hcsr04_ml_simple` L28-29、`dht11_ml` L28、`dht11_ml_simple` L26、`mpu6050_ml` L33/L39、`mpu6050_ml_simple` L30/L32 |
| 2 | 移动平均用 `mode='same'` 零填充，**序列起点被显著拉低**（DHT11 原始温度 19.9 ℃ 对应移动平均值 11.66 ℃） | 各预处理脚本的 `moving_average` |
| 3 | **`dht11_ml_simple.py` 存在变量作用域错误**：`visualize_and_save()` 内引用了只在 `data_split_and_scale()` 中定义的 `X_train/y_train/y_test/X_test`，执行到该处会 `NameError` | `dht11_ml_simple.py` L252-258 |
| 4 | 弃用 API：`fillna(method='bfill')`（pandas ≥2.2 弃用）、`boxplot(labels=...)`（matplotlib ≥3.9 弃用） | `hcsr04_ml` L94、`dht11_ml` L82、`hcsr04_preprocessing` L296 |
| 5 | 图与数据不一致：`set_ylim(5, 18)` 与实测 47.6~102.8 cm 矛盾；"真实值"曲线画的是清洗后均值 | `hcsr04_preprocessing` L275、L269/L286 |
| 6 | 报告中的"采集时长 5 分钟 / 采样频率 1Hz"是脚本内的**硬编码文案**，与实际样本数（DHT11 816、HC-SR04 490）矛盾 | 各预处理脚本的 txt 输出模板 |
| 7 | 未使用导入：`stats`、`PolynomialFeatures`、`PCA`、`cross_val_score`、`StackingRegressor` 等 | `mpu6050_preprocessing` L12、`hcsr04_ml` L11/L18、`mpu6050_ml` 等 |
| 8 | 重复代码：`OutlierDetector` 三份近似复制、`SmoothingFilter` 两份、特征工程逻辑在 6 个 ML 脚本中重复 | 见各文件 |
| 9 | 被注释掉的仿真数据旧代码仍留在文件中（改造痕迹） | `dht11_preprocessing` L46-87、`hcsr04_preprocessing` L44-72 |

> 这些问题是**有意保留并公开**的：它们记录了一次真实课程实验从"跑仿真"到"接真实数据"的改造过程，也说明指标口径（第 5、6 条）和边界效应（第 2 条）是这类数据处理作业最容易踩的坑。

## 7. 能力映射（这段经历能证明什么）

- **传感器与串口**：三类传感器的串口时序参数配置（115200/38400、8N1）、数据完整性验证
- **信号处理**：异常检测（Z-Score/IQR/3-Sigma）、插值修复、四类滤波器的实现与对比、卡尔曼参数（Q/R）调校、互补滤波与姿态解算
- **数据分析工程**：Pandas 数据清洗（GBK 编码、单位字符串处理）、9 宫格可视化、Excel 报告自动生成
- **机器学习流程**：特征工程（滚动统计/滞后/周期编码）、特征选择、多模型对比、5 折交叉验证、模型融合
- **工程反思**：能指出"参考基准取自身均值导致指标失真""零偏估计的静止假设不成立"这类方法论问题
