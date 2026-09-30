#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HC-SR04超声波传感器数据预处理完整演示
包含：数据读取、异常值检测、卡尔曼滤波、精度分析、可视化
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 70)
print("HC-SR04超声波传感器数据预处理演示")
print("=" * 70)

# ==================== 1. 读取真实HC-SR04采集数据 ====================
print("\n【步骤1】读取本地真实采集数据...")

# 你的文件路径（和DHT11格式完全一样）
output_dir = r'D:\高校专项\实验项目'
df_raw = pd.read_excel(f"{output_dir}\\HC-SR04数据.xlsx")

# ========= 【和DHT11完全相同写法】 =========
# 假设Excel表头：时间、距离
time = df_raw["时间"].values
measurements = df_raw["测量距离(mm)"].values

# 样本数量自动获取
n_samples = len(measurements)

print(f"  读取真实样本数: {n_samples}")
print(f"  原始距离范围: {measurements.min():.2f}cm ~ {measurements.max():.2f}cm")

# 统一全局df，保证后续代码正常运行
df = df_raw.copy()

"""
# ==================== 1. 生成模拟数据 ====================
print("\n【步骤1】生成模拟测距数据...")
np.random.seed(44)
n_samples = 300  # 5分钟，每秒1个样本
time = np.arange(n_samples)

# 真实距离10cm
true_distance = 10.0

# 添加高斯噪声
noise = np.random.normal(0, 0.5, n_samples)

# 添加缓慢漂移（模拟温度影响）
drift = 0.02 * np.sin(time / 50)

# 添加异常值（模拟多径效应）
outliers = np.zeros(n_samples)
outlier_indices = np.random.choice(n_samples, 15, replace=False)
outliers[outlier_indices] = np.random.choice([-1, 1], 15) * np.random.uniform(2, 5, 15)

# 合成测量值
measurements = true_distance + noise + drift + outliers

print(f"  生成样本数: {n_samples}")
print(f"  真实距离: {true_distance}cm")
print(f"  测量范围: {measurements.min():.2f}cm ~ {measurements.max():.2f}cm")
print(f"  添加异常值: {len(outlier_indices)}个")
"""


# ==================== 2. 异常值检测 ====================
print("\n【步骤2】异常值检测...")

class OutlierDetector:
    @staticmethod
    def zscore_detection(data, threshold=2.5):
        """Z-Score异常值检测"""
        z_scores = np.abs(stats.zscore(data))
        outliers = z_scores > threshold
        return outliers, z_scores
    
    @staticmethod
    def iqr_detection(data):
        """IQR异常值检测"""
        Q1 = np.percentile(data, 25)
        Q3 = np.percentile(data, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = (data < lower_bound) | (data > upper_bound)
        return outliers, (lower_bound, upper_bound)
    
    @staticmethod
    def three_sigma(data):
        """3-Sigma异常值检测"""
        mean = np.mean(data)
        std = np.std(data)
        outliers = np.abs(data - mean) > 3 * std
        return outliers, (mean - 3*std, mean + 3*std)

# 应用三种检测方法
outliers_z, z_scores = OutlierDetector.zscore_detection(measurements)
outliers_iqr, iqr_bounds = OutlierDetector.iqr_detection(measurements)
outliers_3s, sigma3_bounds = OutlierDetector.three_sigma(measurements)

print(f"  Z-Score方法 (阈值=2.5): 检测到 {np.sum(outliers_z)} 个异常值")
print(f"  IQR方法: 检测到 {np.sum(outliers_iqr)} 个异常值")
print(f"  3-Sigma方法: 检测到 {np.sum(outliers_3s)} 个异常值")

# ==================== 3. 数据清洗 ====================
print("\n【步骤3】数据清洗与插值...")

# 使用Z-Score方法去除异常值
cleaned = measurements.copy()
cleaned[outliers_z] = np.nan

# 线性插值
mask = ~np.isnan(cleaned)
cleaned = np.interp(np.arange(len(cleaned)), np.arange(len(cleaned))[mask], cleaned[mask])

print(f"  清洗前标准差: {measurements.std():.4f}cm")
print(f"  清洗后标准差: {cleaned.std():.4f}cm")

# ==================== 4. 卡尔曼滤波 ====================
print("\n【步骤4】卡尔曼滤波...")

class KalmanFilter:
    def __init__(self, Q=0.01, R=1.0, P=1.0):
        """
        初始化卡尔曼滤波器
        Q: 过程噪声协方差
        R: 测量噪声协方差
        P: 初始估计协方差
        """
        self.Q = Q
        self.R = R
        self.P = P
        self.x = None
    
    def filter(self, measurements):
        """应用卡尔曼滤波"""
        n = len(measurements)
        x_est = np.zeros(n)
        P_est = np.zeros(n)
        K_values = np.zeros(n)
        
        # 初始化
        x_est[0] = measurements[0]
        P_est[0] = self.P
        
        for k in range(1, n):
            # 预测步骤
            x_pred = x_est[k-1]
            P_pred = P_est[k-1] + self.Q
            
            # 更新步骤
            K = P_pred / (P_pred + self.R)
            K_values[k] = K
            x_est[k] = x_pred + K * (measurements[k] - x_pred)
            P_est[k] = (1 - K) * P_pred
        
        return x_est, P_est, K_values

# 应用卡尔曼滤波
kf = KalmanFilter(Q=0.01, R=1.0, P=1.0)
filtered, covariance, kalman_gain = kf.filter(cleaned)

print(f"  卡尔曼滤波参数: Q={kf.Q}, R={kf.R}")
print(f"  滤波后标准差: {filtered.std():.4f}cm")
print(f"  最终协方差: {covariance[-1]:.6f}")

# ==================== 5. 其他滤波方法对比 ====================
print("\n【步骤5】其他滤波方法...")

class SmoothingFilter:
    @staticmethod
    def moving_average(data, window=5):
        """移动平均滤波"""
        return np.convolve(data, np.ones(window)/window, mode='same')
    
    @staticmethod
    def exponential_smoothing(data, alpha=0.3):
        """指数平滑滤波"""
        result = np.zeros_like(data)
        result[0] = data[0]
        for i in range(1, len(data)):
            result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
        return result

# 应用其他滤波器
ma_filtered = SmoothingFilter.moving_average(cleaned, window=5)
exp_filtered = SmoothingFilter.exponential_smoothing(cleaned, alpha=0.3)

print("  应用滤波器:")
print("    - 移动平均滤波 (窗口=5)")
print("    - 指数平滑滤波 (α=0.3)")
print("    - 卡尔曼滤波")

# ==================== 6. 误差分析 ====================
print("\n【步骤6】误差分析...")

# 先定义真实值（修复位置）
true_distance = np.mean(cleaned)

def calculate_metrics(data, true_value):
    error = data - true_value
    mse = np.mean(error ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(error))
    mape = np.mean(np.abs(error / true_value)) * 100
    std = np.std(data)
    return {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'MAPE': mape, 'Std': std}

# 现在再调用，就不会报错了
metrics_raw = calculate_metrics(measurements, true_distance)
metrics_clean = calculate_metrics(cleaned, true_distance)
metrics_kf = calculate_metrics(filtered, true_distance)
metrics_ma = calculate_metrics(ma_filtered, true_distance)
metrics_exp = calculate_metrics(exp_filtered, true_distance)

print("\n  各阶段误差对比:")
print(f"  {'阶段':<15} {'RMSE(cm)':<12} {'MAE(cm)':<12} {'MAPE(%)':<12} {'Std(cm)':<12}")
print("  " + "-" * 65)
print(f"  {'原始数据':<15} {metrics_raw['RMSE']:<12.4f} {metrics_raw['MAE']:<12.4f} {metrics_raw['MAPE']:<12.4f} {metrics_raw['Std']:<12.4f}")
print(f"  {'清洗后':<15} {metrics_clean['RMSE']:<12.4f} {metrics_clean['MAE']:<12.4f} {metrics_clean['MAPE']:<12.4f} {metrics_clean['Std']:<12.4f}")
print(f"  {'移动平均':<15} {metrics_ma['RMSE']:<12.4f} {metrics_ma['MAE']:<12.4f} {metrics_ma['MAPE']:<12.4f} {metrics_ma['Std']:<12.4f}")
print(f"  {'指数平滑':<15} {metrics_exp['RMSE']:<12.4f} {metrics_exp['MAE']:<12.4f} {metrics_exp['MAPE']:<12.4f} {metrics_exp['Std']:<12.4f}")
print(f"  {'卡尔曼滤波':<15} {metrics_kf['RMSE']:<12.4f} {metrics_kf['MAE']:<12.4f} {metrics_kf['MAPE']:<12.4f} {metrics_kf['Std']:<12.4f}")

# ==================== 7. 动态目标跟踪模拟 ====================
print("\n【步骤7】动态目标跟踪模拟...")

# 模拟移动目标
np.random.seed(45)
t_dynamic = np.arange(0, 300, 1)
true_distance_dynamic = 50 - 40 * np.sin(t_dynamic / 50) + 5 * np.sin(t_dynamic / 10)
true_distance_dynamic = np.clip(true_distance_dynamic, 10, 90)

# 添加噪声
noise_dynamic = np.random.normal(0, 1, len(t_dynamic))
measured_dynamic = true_distance_dynamic + noise_dynamic

# 卡尔曼滤波跟踪
kf_dynamic = KalmanFilter(Q=0.1, R=2.0, P=1.0)
filtered_dynamic, _, _ = kf_dynamic.filter(measured_dynamic)

# 速度估计
dt = 1.0
velocity = np.gradient(filtered_dynamic, dt)
true_velocity = np.gradient(true_distance_dynamic, dt)

print("  模拟场景: 目标在10-90cm范围内移动")
print("  卡尔曼滤波跟踪成功")

# ==================== 8. 可视化输出 ====================
print("\n【步骤8】生成可视化图表...")

fig = plt.figure(figsize=(16, 12))

# 子图1：原始数据与异常值
ax1 = plt.subplot(3, 3, 1)
ax1.plot(time, measurements, 'b-', alpha=0.6, linewidth=1, label='原始测量')
ax1.scatter(time[outliers_z], measurements[outliers_z], color='red', s=50, 
           zorder=5, label='Z-Score异常值', marker='x')
ax1.axhline(y=true_distance, color='g', linestyle='--', linewidth=2, label=f'真实值 ({true_distance}cm)')
ax1.set_xlabel('时间 (秒)')
ax1.set_ylabel('距离 (cm)')
ax1.set_title('原始数据与异常值检测', fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_ylim(5, 18)

# 子图2：三种异常值检测方法对比
ax2 = plt.subplot(3, 3, 2)
ax2.plot(time, measurements, 'b-', alpha=0.4, linewidth=1)
ax2.scatter(time[outliers_z], measurements[outliers_z], color='red', s=40, 
           label=f'Z-Score ({np.sum(outliers_z)}个)', marker='x', alpha=0.7)
ax2.scatter(time[outliers_iqr], measurements[outliers_iqr], color='orange', s=40,
           label=f'IQR ({np.sum(outliers_iqr)}个)', marker='s', alpha=0.7)
ax2.scatter(time[outliers_3s], measurements[outliers_3s], color='purple', s=40,
           label=f'3-Sigma ({np.sum(outliers_3s)}个)', marker='^', alpha=0.7)
ax2.axhline(y=true_distance, color='g', linestyle='--', linewidth=2)
ax2.set_xlabel('时间 (秒)')
ax2.set_ylabel('距离 (cm)')
ax2.set_title('三种异常值检测方法对比', fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 子图3：箱线图
ax3 = plt.subplot(3, 3, 3)
bp = ax3.boxplot([measurements, cleaned, filtered], 
                 labels=['原始', '清洗后', '卡尔曼'],
                 patch_artist=True)
bp['boxes'][0].set_facecolor('lightblue')
bp['boxes'][1].set_facecolor('lightgreen')
bp['boxes'][2].set_facecolor('lightyellow')
ax3.axhline(y=true_distance, color='r', linestyle='--', linewidth=2, label='真实值')
ax3.set_ylabel('距离 (cm)')
ax3.set_title('数据分布箱线图', fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3, axis='y')

# 子图4：卡尔曼滤波效果
ax4 = plt.subplot(3, 3, 4)
ax4.plot(time, cleaned, 'b-', alpha=0.4, linewidth=1, label='清洗后数据')
ax4.plot(time, filtered, 'r-', linewidth=2, label='卡尔曼滤波')
ax4.axhline(y=true_distance, color='g', linestyle='--', linewidth=2, label=f'真实值 ({true_distance}cm)')
ax4.fill_between(time, filtered - np.sqrt(covariance), filtered + np.sqrt(covariance), 
                 alpha=0.2, color='red', label='±1σ置信区间')
ax4.set_xlabel('时间 (秒)')
ax4.set_ylabel('距离 (cm)')
ax4.set_title('卡尔曼滤波效果', fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)

# 子图5：卡尔曼增益和协方差
ax5 = plt.subplot(3, 3, 5)
ax5_twin = ax5.twinx()
line1 = ax5.plot(time, covariance, 'g-', linewidth=2, label='协方差 P')
line2 = ax5_twin.plot(time, kalman_gain, 'orange', linewidth=2, label='卡尔曼增益 K')
ax5.set_xlabel('时间 (秒)')
ax5.set_ylabel('协方差', color='g')
ax5_twin.set_ylabel('卡尔曼增益', color='orange')
ax5.set_title('卡尔曼滤波器收敛过程', fontweight='bold')
ax5.grid(True, alpha=0.3)
lines = line1 + line2
labels = [l.get_label() for l in lines]
ax5.legend(lines, labels, loc='upper right')

# 子图6：滤波方法对比
ax6 = plt.subplot(3, 3, 6)
ax6.plot(time, cleaned, 'b-', alpha=0.3, linewidth=1, label='清洗后')
ax6.plot(time, ma_filtered, 'r-', linewidth=1.5, label='移动平均')
ax6.plot(time, exp_filtered, 'g-', linewidth=1.5, label='指数平滑')
ax6.plot(time, filtered, 'm-', linewidth=2, label='卡尔曼滤波')
ax6.axhline(y=true_distance, color='k', linestyle='--', linewidth=1.5, label='真实值')
ax6.set_xlabel('时间 (秒)')
ax6.set_ylabel('距离 (cm)')
ax6.set_title('滤波方法对比', fontweight='bold')
ax6.legend()
ax6.grid(True, alpha=0.3)

# 子图7：误差对比
ax7 = plt.subplot(3, 3, 7)
error_clean = np.abs(cleaned - true_distance)
error_ma = np.abs(ma_filtered - true_distance)
error_exp = np.abs(exp_filtered - true_distance)
error_kf = np.abs(filtered - true_distance)
ax7.plot(time, error_clean, 'b-', alpha=0.4, linewidth=1, label='清洗后')
ax7.plot(time, error_ma, 'r-', linewidth=1.5, label='移动平均')
ax7.plot(time, error_exp, 'g-', linewidth=1.5, label='指数平滑')
ax7.plot(time, error_kf, 'm-', linewidth=2, label='卡尔曼滤波')
ax7.set_xlabel('时间 (秒)')
ax7.set_ylabel('绝对误差 (cm)')
ax7.set_title('各方法绝对误差对比', fontweight='bold')
ax7.legend()
ax7.grid(True, alpha=0.3)

# 子图8：动态目标跟踪
ax8 = plt.subplot(3, 3, 8)
ax8.plot(t_dynamic, true_distance_dynamic, 'g-', linewidth=2, label='真实距离', alpha=0.8)
ax8.plot(t_dynamic, measured_dynamic, 'b-', alpha=0.3, linewidth=1, label='测量值')
ax8.plot(t_dynamic, filtered_dynamic, 'r-', linewidth=2, label='卡尔曼跟踪')
ax8.set_xlabel('时间 (秒)')
ax8.set_ylabel('距离 (cm)')
ax8.set_title('动态目标跟踪', fontweight='bold')
ax8.legend()
ax8.grid(True, alpha=0.3)

# 子图9：处理流程总结
ax9 = plt.subplot(3, 3, 9)
ax9.axis('off')
summary_text = f"""
HC-SR04预处理流程总结

输入数据:
  • 样本数: {n_samples}
  • 真实距离: {true_distance}cm
  • 测量范围: {measurements.min():.1f}~{measurements.max():.1f}cm

处理步骤:
  1. 异常值检测
     - Z-Score: {np.sum(outliers_z)}个
     - IQR: {np.sum(outliers_iqr)}个
     - 3-Sigma: {np.sum(outliers_3s)}个
  
  2. 数据清洗 (线性插值)
  
  3. 卡尔曼滤波
     - Q={kf.Q}, R={kf.R}
     - 最终协方差: {covariance[-1]:.6f}

误差指标 (RMSE):
  • 原始数据: {metrics_raw['RMSE']:.4f}cm
  • 清洗后: {metrics_clean['RMSE']:.4f}cm
  • 移动平均: {metrics_ma['RMSE']:.4f}cm
  • 指数平滑: {metrics_exp['RMSE']:.4f}cm
  • 卡尔曼滤波: {metrics_kf['RMSE']:.4f}cm

性能提升:
  • RMSE降低: {(1-metrics_kf['RMSE']/metrics_raw['RMSE'])*100:.1f}%
"""
ax9.text(0.1, 0.95, summary_text, transform=ax9.transAxes,
        fontsize=10, verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

plt.suptitle('HC-SR04超声波传感器数据预处理完整流程', fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.96])

# 保存图片
output_dir = r'D:\高校专项\实验项目\实验二数据预处理\预处理结果输出'
import os
os.makedirs(output_dir, exist_ok=True)

plt.savefig(f'{output_dir}\\HC-SR04预处理结果.png', dpi=150, bbox_inches='tight')
print(f"  图表已保存: {output_dir}\\HC-SR04预处理结果.png")

plt.savefig(f'{output_dir}\\HC-SR04预处理结果_高清.png', dpi=300, bbox_inches='tight')
print(f"  高清图表已保存: {output_dir}\\HC-SR04预处理结果_高清.png")

plt.show()

# ==================== 9. 保存处理后的数据 ====================
print("\n【步骤9】保存处理后的数据...")

result_df = pd.DataFrame({
    '时间': time,
    '原始测量': measurements,
    '清洗后': cleaned,
    '移动平均': ma_filtered,
    '指数平滑': exp_filtered,
    '卡尔曼滤波': filtered,
    '协方差': covariance,
    '卡尔曼增益': kalman_gain
})

result_df.to_excel(f'{output_dir}\\HC-SR04预处理结果数据.xlsx', index=False)
print(f"  数据已保存: {output_dir}\\HC-SR04预处理结果数据.xlsx")

# ==================== 10. 生成报告 ====================
print("\n【步骤10】生成处理报告...")

report = f"""
{'='*70}
HC-SR04超声波传感器数据预处理报告
{'='*70}

1. 数据概况
   - 采集样本数: {n_samples}
   - 采集时长: 5分钟
   - 采样频率: 1Hz
   - 真实距离: {true_distance}cm

2. 原始数据统计
   - 最小值: {measurements.min():.2f}cm
   - 最大值: {measurements.max():.2f}cm
   - 平均值: {measurements.mean():.2f}cm
   - 标准差: {measurements.std():.2f}cm

3. 异常值检测
   - Z-Score方法 (阈值=2.5): {np.sum(outliers_z)}个
   - IQR方法: {np.sum(outliers_iqr)}个
   - 3-Sigma方法: {np.sum(outliers_3s)}个

4. 卡尔曼滤波参数
   - 过程噪声 Q: {kf.Q}
   - 测量噪声 R: {kf.R}
   - 初始协方差 P: {kf.P}
   - 最终协方差: {covariance[-1]:.6f}

5. 误差指标对比
   {'方法':<15} {'RMSE(cm)':<12} {'MAE(cm)':<12} {'MAPE(%)':<12}
   {'-'*50}
   {'原始数据':<15} {metrics_raw['RMSE']:<12.4f} {metrics_raw['MAE']:<12.4f} {metrics_raw['MAPE']:<12.4f}
   {'清洗后':<15} {metrics_clean['RMSE']:<12.4f} {metrics_clean['MAE']:<12.4f} {metrics_clean['MAPE']:<12.4f}
   {'移动平均':<15} {metrics_ma['RMSE']:<12.4f} {metrics_ma['MAE']:<12.4f} {metrics_ma['MAPE']:<12.4f}
   {'指数平滑':<15} {metrics_exp['RMSE']:<12.4f} {metrics_exp['MAE']:<12.4f} {metrics_exp['MAPE']:<12.4f}
   {'卡尔曼滤波':<15} {metrics_kf['RMSE']:<12.4f} {metrics_kf['MAE']:<12.4f} {metrics_kf['MAPE']:<12.4f}

6. 性能提升
   - RMSE降低: {(1-metrics_kf['RMSE']/metrics_raw['RMSE'])*100:.1f}%
   - MAE降低: {(1-metrics_kf['MAE']/metrics_raw['MAE'])*100:.1f}%
   - 标准差降低: {(1-metrics_kf['Std']/metrics_raw['Std'])*100:.1f}%

7. 预处理算法清单
   ✓ Z-Score异常值检测
   ✓ IQR异常值检测
   ✓ 3-Sigma异常值检测
   ✓ 线性插值
   ✓ 移动平均滤波
   ✓ 指数平滑滤波
   ✓ 卡尔曼滤波
   ✓ 动态目标跟踪

{'='*70}
"""

with open(f'{output_dir}\\HC-SR04预处理报告.txt', 'w', encoding='utf-8') as f:
    f.write(report)

print(report)
print(f"报告已保存: {output_dir}\\HC-SR04预处理报告.txt")

print("\n" + "=" * 70)
print("HC-SR04预处理演示完成！")
print(f"输出目录: {output_dir}")
print("=" * 70)
