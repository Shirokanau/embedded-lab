# 传感器数据采集与分析

课程实验（传感器实验）的整理。三种传感器（DHT11 温湿度、HC-SR04 超声波、MPU-6050 六轴）的串口数据采集、异常检测与滤波、姿态解算和回归建模，用 Python 完成。

| 脚本 | 数据来源 |
|---|---|
| `src/preprocessing/hcsr04_preprocessing_demo.py` | 读实测 Excel 的 `时间`、`测量距离(mm)` 两列 |
| `src/preprocessing/dht11_preprocessing_demo.py` | 读实测 Excel 的 `时间`、`温度`、`湿度` 三列 |
| `src/preprocessing/mpu6050_preprocessing_demo.py` | 脚本内生成仿真数据（`seed=46`，600 点 @20Hz），不读外部文件 |
| `src/machine_learning/hcsr04_machine_learning.py` | 优先读 Excel，文件不存在时走脚本内仿真分支 |
| `src/machine_learning/mpu6050_machine_learning.py` | 同上 |
| `src/machine_learning/dht11_machine_learning.py` | 脚本内生成仿真数据（`seed=42`，300 条） |
| `src/machine_learning/dht11_ml_simple.py` | 脚本内生成仿真数据 |
| `src/machine_learning/hcsr04_ml_simple.py` | 只读 Excel，无仿真回退 |
| `src/machine_learning/mpu6050_ml_simple.py` | 只读 Excel，无仿真回退 |

因此 `reports/MPU-6050预处理报告.txt` 里的指标是仿真数据的结果；DHT11 和 HC-SR04 的报告来自实测数据。原始采集的 Excel 没有保留下来（原本存在实验机上），仓库里只有脚本、脚本输出的数据表和报告，所以部分脚本不能直接跑通，见文末。

## 实验一：串口采集

| 项 | 内容 |
|---|---|
| 传感器 | DHT11 温湿度、HC-SR04 超声波、MPU-6050 六轴姿态 |
| 串口参数 | 温湿度与姿态 115200，超声波 38400；8 数据位、1 停止位、无校验 |
| 采集结果 | 湿度 57.0%~95.0% RH，温度 25.9~26.6 ℃；测距 0~3926 mm；姿态 Roll/Pitch/Yaw |
| 小结 | 通信稳定无乱码，波特率要和传感器对上 |

## 实验二：预处理与滤波

读数据 → 异常检测 → 置空后插值补回 → 滤波 → 算误差指标 → 出图并导出 Excel 和 txt 报告。

### 参数

| 环节 | 方法 | 参数 |
|---|---|---|
| 异常检测 | Z-Score | 阈值 2.5 |
| | IQR | 系数 1.5 |
| | 3-Sigma | 阈值 3 |
| 插值 | `interpolate(linear, limit_direction='both')` / `np.interp` | |
| 滤波 | 移动平均 | 窗口 5，`np.convolve(..., mode='same')` |
| | 指数平滑 | α = 0.3 |
| | Savitzky-Golay | 窗口 7，阶数 3 |
| | 卡尔曼（自己写的） | Q=0.01，R=1.0，P=1.0 |
| 姿态解算（MPU-6050，仿真数据） | 陀螺仪零偏校准 | 取前 100 个样本的均值 |
| | Butterworth 滤波 | 加速度计低通 5Hz、陀螺仪高通 0.5Hz，fs=20Hz，order=2，filtfilt |
| | 互补滤波 | α = 0.96 |
| 标准化 | Z-Score / Min-Max / Robust | 三种都写了，实际只用了 Z-Score |

### HC-SR04（实测，490 组）

| 处理方式 | RMSE (cm) | MAE (cm) | MAPE (%) |
|---|---|---|---|
| 原始数据 | 3.6574 | 1.5143 | 1.5154 |
| 清洗后 | 1.4709 | 1.2779 | 1.2788 |
| 移动平均 | 2.8595 | 0.7552 | 0.7558 |
| 指数平滑 | 0.5997 | 0.4843 | 0.4846 |
| 卡尔曼滤波 | 0.3076 | 0.2377 | 0.2379 |

脚本里算 RMSE 用的"真实值"是**清洗后数据的均值**（`np.mean(cleaned)`，见 `hcsr04_preprocessing_demo.py` 第 207 行），不是另外测出来的真值。滤波后的曲线本来就离自己的均值更近，所以这里的 RMSE 下降只能说明平滑效果，不能当成测距精度提高了多少。

### DHT11（实测，816 组）

- 温度 18.3~21.7 ℃，均值 20.01 ℃，标准差 0.88 ℃；湿度 35.1%~44.9%，均值 39.99%，标准差 2.41%
- Z-Score 检测到的异常值：温度和湿度都是 0 个
- 滤波后标准差：温度用 SG 滤波 0.5053 ℃，湿度用指数平滑 1.0024%
- 温湿度相关系数 0.0405

### MPU-6050（仿真数据，600 组 @20Hz）

| 解算方式 | 俯仰角 RMSE (°) | 横滚角 RMSE (°) |
|---|---|---|
| 纯加速度计 | 0.722 | 0.773 |
| 纯陀螺仪 | 11.175 | 7.062 |
| 互补滤波（α=0.96） | 5.211 | 4.873 |

报告里"性能提升"直接打印成 -621.8% / -530.6%，也就是互补滤波比纯加速度计还差。原因有三个：零偏校准假设前 100 个样本是静止的，但这段数据是 Figure-8 动态运动，零偏估计偏了很多（估计 X=2.914°/s，实际 0.300°/s）；仿真数据里加速度计几乎没噪声，静态解算本来就占优；互补滤波只用了 gx/gy，没用 gz。

## 实验三：建模

特征工程：时间（hour/minute/周期编码）、滚动均值与标准差（窗口 5/10/20）、EMA、1/3/5 阶滞后、加速度与角速度模、姿态估计、温湿度交互等；再用 SelectKBest（mutual_info_regression 或 f_regression，k=15）筛特征，StandardScaler 标准化。

模型：LinearRegression、Ridge、Lasso、ElasticNet、RandomForest(100)、GradientBoosting(100)、SVR(rbf)、MLP。随机森林用 GridSearchCV(cv=5, scoring='r2') 调参；评估用 R²/RMSE/MAE 加 5 折交叉验证；融合用 Top-3 简单平均、按测试集 R² 加权的加权平均，以及 VotingRegressor。

DHT11 的两个建模脚本用的是脚本内仿真数据；`hcsr04_ml_simple.py` 和 `mpu6050_ml_simple.py` 需要先有预处理输出的 Excel（对应文件没有找到）。

## 目录结构

```
02-sensor-data-analysis/
├── src/
│   ├── preprocessing/            # 实验二，3 个脚本
│   └── machine_learning/         # 实验三，完整版和简化版各 3 个
├── data/
│   ├── preprocessing/            # 预处理输出数据，保留原始测量列
│   └── machine_learning/         # 模型性能对比、预测结果等
├── reports/                      # 三个传感器各一份 txt 报告
└── results/                      # 结果图
```

## 运行环境与数据依赖

依赖见 `requirements.txt`：numpy、pandas、matplotlib、scipy、scikit-learn、openpyxl。

脚本里设了 `font.sans-serif = ['SimHei', ...]`，非 Windows 环境需要自备中文字体，否则图里的中文会是方框。

脚本里的数据路径是写死的绝对路径（原实验机的 `D:\高校专项\实验项目\...`），换机器要改。原本要读的 4 个 Excel 都不在仓库里：

| 文件 | 需要的列 |
|---|---|
| `HC-SR04数据.xlsx` | `时间`、`测量距离(mm)` |
| `第2组温湿度校验数据-1010.xlsx` | `时间`、`温度`、`湿度` |
| `HC-SR04预处理结果数据.xlsx` | `时间`、`距离`、`速度`、`加速度`、`真实距离` |
| `MPU-6050预处理结果数据.xlsx` | `时间`、`Ax..Gz`、`真实俯仰角`、`真实横滚角` |

不用外部数据也能跑的：`mpu6050_preprocessing_demo.py`（纯仿真）、`dht11_machine_learning.py`，以及两个带仿真回退的建模脚本。`data/` 里的预处理结果表保留了原始测量列，可以直接看真实数据和滤波结果的对照。

## 已知问题

代码保持原样，没有修改。

| # | 问题 | 位置 |
|---|---|---|
| 1 | 数据路径写死成实验机的绝对路径，共 10 处 | 各脚本开头与输出段 |
| 2 | 移动平均用 `mode='same'` 零填充，序列开头被拉低（DHT11 原始温度 19.9 ℃ 对应的移动平均是 11.66 ℃） | 各预处理脚本的 `moving_average` |
| 3 | `visualize_and_save()` 里引用了只在另一个函数里定义的 `X_train/y_train/y_test/X_test`，跑到这里会 NameError | `dht11_ml_simple.py` 第 252-258 行 |
| 4 | 用了已弃用的 API：`fillna(method='bfill')`、`boxplot(labels=...)` | `hcsr04_ml` L94、`dht11_ml` L82、`hcsr04_preprocessing` L296 |
| 5 | 图与数据不一致：`set_ylim(5, 18)` 和实测的 47.6~102.8 cm 对不上；"真实值"曲线画的是清洗后均值 | `hcsr04_preprocessing` L275、L269/L286 |
| 6 | 报告里"采集时长 5 分钟 / 采样频率 1Hz"是模板里写死的文案，和实际样本数（DHT11 816、HC-SR04 490）对不上 | 各预处理脚本的 txt 输出模板 |
| 7 | 没用到的导入：`stats`、`PolynomialFeatures`、`PCA`、`StackingRegressor` 等 | 多个脚本 |
| 8 | 重复代码：`OutlierDetector` 写了三份，`SmoothingFilter` 两份，特征工程逻辑在 6 个建模脚本里各写了一遍 | 见各文件 |
| 9 | 改造时注释掉的仿真数据代码还留在文件里 | `dht11_preprocessing` L46-87、`hcsr04_preprocessing` L44-72 |
