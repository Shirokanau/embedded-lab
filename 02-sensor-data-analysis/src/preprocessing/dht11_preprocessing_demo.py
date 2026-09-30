#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DHT11温湿度传感器数据预处理完整演示
包含：数据读取、清洗、异常值检测、滤波、标准化、可视化
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.signal import savgol_filter
import warnings

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 70)
print("DHT11温湿度传感器数据预处理演示")
print("=" * 70)
# ==================== 1. 读取真实DHT11采集数据 ====================
print("\n【步骤1】读取本地真实采集数据...")

# 你的文件路径
output_dir = r'D:\高校专项\实验项目'
df_raw = pd.read_excel(f"{output_dir}\\第2组温湿度校验数据-1010.xlsx")

# ========= 重点：根据你Excel里的列名修改 =========
# 假设你的Excel表头为：时间、温度、湿度
time = df_raw["时间"].values
temperature = df_raw["温度"].values
humidity = df_raw["湿度"].values

# 样本数量自动获取
n_samples = len(temperature)

print(f"  读取真实样本数: {n_samples}")
print(f"  原始温度范围: {temperature.min():.2f}°C ~ {temperature.max():.2f}°C")
print(f"  原始湿度范围: {humidity.min():.2f}% ~ {humidity.max():.2f}%")

# 统一全局df，保证后续代码正常运行
df = df_raw.copy()
"""
# ==================== 1. 生成模拟数据 ====================
print("\n【步骤1】生成模拟采集数据...")
np.random.seed(42)
n_samples = 300  # 5分钟，每秒1个样本

# 时间轴
time = np.arange(n_samples)

# 温度数据：哈尔滨室温约20°C，加上趋势和噪声
temp_base = 20.0
temp_trend = 0.5 * np.sin(time / 50)  # 缓慢变化趋势
temp_noise = np.random.normal(0, 0.5, n_samples)
temperature = temp_base + temp_trend + temp_noise

# 添加异常值
temp_outliers_idx = np.random.choice(n_samples, 10, replace=False)
temperature[temp_outliers_idx] += np.random.choice([-1, 1], 10) * np.random.uniform(2, 4, 10)

# 湿度数据：相对湿度约55%，与温度负相关
humidity_base = 55.0
humidity_corr = -0.3 * (temperature - temp_base)  # 与温度负相关
humidity_noise = np.random.normal(0, 1.5, n_samples)
humidity = humidity_base + humidity_corr + humidity_noise
humidity = np.clip(humidity, 20, 90)  # 限制在合理范围

# 添加异常值
hum_outliers_idx = np.random.choice(n_samples, 8, replace=False)
humidity[hum_outliers_idx] += np.random.choice([-1, 1], 8) * np.random.uniform(5, 10, 8)
humidity = np.clip(humidity, 20, 90)

print(f"  生成样本数: {n_samples}")
print(f"  温度范围: {temperature.min():.2f}°C ~ {temperature.max():.2f}°C")
print(f"  湿度范围: {humidity.min():.2f}% ~ {humidity.max():.2f}%")

# 创建DataFrame
df = pd.DataFrame({
    '时间': time,
    '温度': temperature,
    '湿度': humidity
})
"""
# ==================== 2. 异常值检测 ====================
print("\n【步骤2】异常值检测（Z-Score方法）...")


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


# 温度异常值检测
temp_outliers_z, temp_zscores = OutlierDetector.zscore_detection(temperature)
temp_outliers_iqr, temp_bounds = OutlierDetector.iqr_detection(temperature)

# 湿度异常值检测
hum_outliers_z, hum_zscores = OutlierDetector.zscore_detection(humidity)
hum_outliers_iqr, hum_bounds = OutlierDetector.iqr_detection(humidity)

print(f"  温度异常值(Z-Score): {np.sum(temp_outliers_z)} 个")
print(f"  温度异常值(IQR): {np.sum(temp_outliers_iqr)} 个")
print(f"  湿度异常值(Z-Score): {np.sum(hum_outliers_z)} 个")
print(f"  湿度异常值(IQR): {np.sum(hum_outliers_iqr)} 个")

# ==================== 3. 数据清洗 ====================
print("\n【步骤3】数据清洗（去除异常值并插值）...")

# 使用Z-Score方法去除异常值
temp_clean = temperature.copy()
temp_clean[temp_outliers_z] = np.nan

hum_clean = humidity.copy()
hum_clean[hum_outliers_z] = np.nan

# 线性插值（允许双向插值，避免首尾NaN导致全为NaN）
temp_clean_series = pd.Series(temp_clean).interpolate(method='linear', limit_direction='both')
hum_clean_series = pd.Series(hum_clean).interpolate(method='linear', limit_direction='both')

# 如果插值后仍有NaN（极端情况），用均值填充
if temp_clean_series.isna().any():
    temp_clean_series = temp_clean_series.fillna(np.nanmean(temperature))
if hum_clean_series.isna().any():
    hum_clean_series = hum_clean_series.fillna(np.nanmean(humidity))

temp_clean = temp_clean_series.values
hum_clean = hum_clean_series.values

# 打印清洗后范围（处理NaN情况）
temp_min, temp_max = np.nanmin(temp_clean), np.nanmax(temp_clean)
hum_min, hum_max = np.nanmin(hum_clean), np.nanmax(hum_clean)
print(f"  温度清洗后范围: {temp_min:.2f}°C ~ {temp_max:.2f}°C")
print(f"  湿度清洗后范围: {hum_min:.2f}% ~ {hum_max:.2f}%")

# ==================== 4. 滤波处理 ====================
print("\n【步骤4】应用平滑滤波...")


class SmoothingFilter:
    @staticmethod
    def moving_average(data, window=5):
        """移动平均滤波"""
        return np.convolve(data, np.ones(window) / window, mode='same')

    @staticmethod
    def exponential_smoothing(data, alpha=0.3):
        """指数平滑滤波"""
        result = np.zeros_like(data)
        result[0] = data[0]
        for i in range(1, len(data)):
            result[i] = alpha * data[i] + (1 - alpha) * result[i - 1]
        return result

    @staticmethod
    def savgol_filter(data, window=7, polyorder=3):
        """Savitzky-Golay滤波"""
        return savgol_filter(data, window, polyorder)


# 应用滤波
temp_ma = SmoothingFilter.moving_average(temp_clean, window=5)
temp_exp = SmoothingFilter.exponential_smoothing(temp_clean, alpha=0.3)
temp_sg = SmoothingFilter.savgol_filter(temp_clean, window=7, polyorder=3)

hum_ma = SmoothingFilter.moving_average(hum_clean, window=5)
hum_exp = SmoothingFilter.exponential_smoothing(hum_clean, alpha=0.3)

print("  应用滤波器:")
print("    - 移动平均滤波 (窗口=5)")
print("    - 指数平滑滤波 (α=0.3)")
print("    - Savitzky-Golay滤波 (窗口=7, 阶数=3)")

# ==================== 5. 标准化处理 ====================
print("\n【步骤5】数据标准化...")


class Standardization:
    @staticmethod
    def zscore(data):
        """Z-Score标准化"""
        # 确保数据无NaN
        data_clean = data[~np.isnan(data)]
        if len(data_clean) == 0:
            return np.zeros_like(data)
        return (data - np.mean(data_clean)) / np.std(data_clean)

    @staticmethod
    def minmax(data):
        """Min-Max归一化"""
        data_clean = data[~np.isnan(data)]
        if len(data_clean) == 0:
            return np.zeros_like(data)
        return (data - np.min(data_clean)) / (np.max(data_clean) - np.min(data_clean))

    @staticmethod
    def robust(data):
        """Robust标准化"""
        data_clean = data[~np.isnan(data)]
        if len(data_clean) == 0:
            return np.zeros_like(data)
        median = np.median(data_clean)
        mad = np.median(np.abs(data_clean - median))
        if mad == 0:
            return np.zeros_like(data)
        return (data - median) / mad


# 使用Savitzky-Golay滤波后的数据进行标准化
temp_zscore = Standardization.zscore(temp_sg)
hum_zscore = Standardization.zscore(hum_exp)

print(f"  温度标准化后: 均值={np.nanmean(temp_zscore):.4f}, 标准差={np.nanstd(temp_zscore):.4f}")
print(f"  湿度标准化后: 均值={np.nanmean(hum_zscore):.4f}, 标准差={np.nanstd(hum_zscore):.4f}")

# ==================== 6. 误差分析 ====================
print("\n【步骤6】误差分析...")


# 计算各阶段误差指标
def calculate_metrics(original, processed):
    """计算误差指标（忽略NaN）"""
    mask = ~(np.isnan(original) | np.isnan(processed))
    if mask.sum() == 0:
        return {'MSE': np.nan, 'RMSE': np.nan, 'MAE': np.nan, 'Std': np.nan}
    orig = original[mask]
    proc = processed[mask]
    mse = np.mean((orig - proc) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(orig - proc))
    std = np.std(proc)
    return {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'Std': std}


# 以清洗后的数据为参考，计算滤波前后的差异
temp_metrics_before = calculate_metrics(temp_clean, temp_clean)
temp_metrics_after = calculate_metrics(temp_clean, temp_sg)

print("  温度数据误差指标:")
print(f"    原始数据标准差: {np.std(temperature):.4f}")
print(f"    清洗后标准差: {np.std(temp_clean):.4f}")
print(f"    滤波后标准差: {np.std(temp_sg):.4f}")

# ==================== 7. 可视化输出 ====================
print("\n【步骤7】生成可视化图表...")

fig = plt.figure(figsize=(16, 12))

# 子图1：原始数据
ax1 = plt.subplot(3, 3, 1)
ax1.plot(time, temperature, 'b-', alpha=0.7, linewidth=1, label='温度')
ax1_twin = ax1.twinx()
ax1_twin.plot(time, humidity, 'r-', alpha=0.7, linewidth=1, label='湿度')
ax1.set_xlabel('时间 (秒)')
ax1.set_ylabel('温度 (°C)', color='b')
ax1_twin.set_ylabel('湿度 (%)', color='r')
ax1.set_title('原始数据', fontweight='bold')
ax1.grid(True, alpha=0.3)

# 子图2：异常值检测
ax2 = plt.subplot(3, 3, 2)
ax2.plot(time, temperature, 'b-', alpha=0.5, linewidth=1, label='原始温度')
ax2.scatter(time[temp_outliers_z], temperature[temp_outliers_z],
            color='red', s=50, zorder=5, label='Z-Score异常值', marker='x')
ax2.set_xlabel('时间 (秒)')
ax2.set_ylabel('温度 (°C)')
ax2.set_title('异常值检测 (Z-Score)', fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 子图3：清洗后数据
ax3 = plt.subplot(3, 3, 3)
ax3.plot(time, temp_clean, 'g-', linewidth=1.5, label='清洗后温度')
ax3.plot(time, hum_clean, 'm-', linewidth=1.5, label='清洗后湿度')
ax3.set_xlabel('时间 (秒)')
ax3.set_ylabel('数值')
ax3.set_title('数据清洗后', fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 子图4：滤波对比 - 温度
ax4 = plt.subplot(3, 3, 4)
ax4.plot(time, temp_clean, 'b-', alpha=0.4, linewidth=1, label='清洗后')
ax4.plot(time, temp_ma, 'r-', linewidth=2, label='移动平均')
ax4.plot(time, temp_exp, 'g-', linewidth=2, label='指数平滑')
ax4.plot(time, temp_sg, 'm-', linewidth=2, label='Savitzky-Golay')
ax4.set_xlabel('时间 (秒)')
ax4.set_ylabel('温度 (°C)')
ax4.set_title('温度滤波方法对比', fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)

# 子图5：滤波对比 - 湿度
ax5 = plt.subplot(3, 3, 5)
ax5.plot(time, hum_clean, 'b-', alpha=0.4, linewidth=1, label='清洗后')
ax5.plot(time, hum_ma, 'r-', linewidth=2, label='移动平均')
ax5.plot(time, hum_exp, 'g-', linewidth=2, label='指数平滑')
ax5.set_xlabel('时间 (秒)')
ax5.set_ylabel('湿度 (%)')
ax5.set_title('湿度滤波方法对比', fontweight='bold')
ax5.legend()
ax5.grid(True, alpha=0.3)

# 子图6：标准化结果
ax6 = plt.subplot(3, 3, 6)
ax6.plot(time, temp_zscore, 'b-', linewidth=1.5, label='温度 (Z-Score)')
ax6.plot(time, hum_zscore, 'r-', linewidth=1.5, label='湿度 (Z-Score)')
ax6.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5)
ax6.set_xlabel('时间 (秒)')
ax6.set_ylabel('标准化值')
ax6.set_title('Z-Score标准化结果', fontweight='bold')
ax6.legend()
ax6.grid(True, alpha=0.3)

# 子图7：误差分布直方图
ax7 = plt.subplot(3, 3, 7)
errors = temperature - temp_clean
# 过滤NaN
errors_valid = errors[~np.isnan(errors)]
ax7.hist(errors_valid, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
ax7.axvline(x=0, color='r', linestyle='--', linewidth=2)
ax7.set_xlabel('误差 (°C)')
ax7.set_ylabel('频数')
ax7.set_title('温度误差分布', fontweight='bold')
ax7.grid(True, alpha=0.3, axis='y')

# 子图8：温湿度相关性
ax8 = plt.subplot(3, 3, 8)
# 过滤NaN
mask_corr = ~(np.isnan(temp_clean) | np.isnan(hum_clean))
temp_clean_valid = temp_clean[mask_corr]
hum_clean_valid = hum_clean[mask_corr]
ax8.scatter(temp_clean_valid, hum_clean_valid, alpha=0.5, s=20)
ax8.set_xlabel('温度 (°C)')
ax8.set_ylabel('湿度 (%)')
if len(temp_clean_valid) > 1:
    correlation = np.corrcoef(temp_clean_valid, hum_clean_valid)[0, 1]
else:
    correlation = np.nan
ax8.set_title(f'温湿度相关性 (r={correlation:.3f})', fontweight='bold')
ax8.grid(True, alpha=0.3)

# 添加趋势线（仅当有效数据点≥2）
if len(temp_clean_valid) >= 2:
    try:
        z = np.polyfit(temp_clean_valid, hum_clean_valid, 1)
        p = np.poly1d(z)
        ax8.plot(temp_clean_valid, p(temp_clean_valid), "r--", alpha=0.8, linewidth=2)
    except np.linalg.LinAlgError:
        print("  警告：多项式拟合失败，跳过趋势线")

# 子图9：处理流程总结
ax9 = plt.subplot(3, 3, 9)
ax9.axis('off')
summary_text = f"""
DHT11预处理流程总结

输入数据:
  • 样本数: {n_samples}
  • 温度范围: {temperature.min():.1f}~{temperature.max():.1f}°C
  • 湿度范围: {humidity.min():.1f}~{humidity.max():.1f}%

处理步骤:
  1. 异常值检测 (Z-Score)
     - 温度异常: {np.sum(temp_outliers_z)}个
     - 湿度异常: {np.sum(hum_outliers_z)}个

  2. 数据清洗 (线性插值)

  3. 平滑滤波
     - 移动平均 (N=5)
     - 指数平滑 (α=0.3)
     - Savitzky-Golay

  4. Z-Score标准化

输出指标:
  • 温度标准差: {np.nanstd(temp_sg):.3f}°C
  • 湿度标准差: {np.nanstd(hum_exp):.3f}%
  • 温湿度相关系数: {correlation:.3f}
"""
ax9.text(0.1, 0.95, summary_text, transform=ax9.transAxes,
         fontsize=10, verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.suptitle('DHT11温湿度传感器数据预处理完整流程', fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.96])

# 保存图片
output_dir = r'D:\高校专项\实验项目\实验二数据预处理\预处理结果输出'
import os

os.makedirs(output_dir, exist_ok=True)

plt.savefig(f'{output_dir}\\DHT11预处理结果.png', dpi=150, bbox_inches='tight')
print(f"  图表已保存: {output_dir}\\DHT11预处理结果.png")

plt.savefig(f'{output_dir}\\DHT11预处理结果_高清.png', dpi=300, bbox_inches='tight')
print(f"  高清图表已保存: {output_dir}\\DHT11预处理结果_高清.png")

plt.show()

# ==================== 8. 保存处理后的数据 ====================
print("\n【步骤8】保存处理后的数据...")

result_df = pd.DataFrame({
    '时间': time,
    '原始温度': temperature,
    '原始湿度': humidity,
    '清洗后温度': temp_clean,
    '清洗后湿度': hum_clean,
    '温度_移动平均': temp_ma,
    '温度_指数平滑': temp_exp,
    '温度_SG滤波': temp_sg,
    '湿度_移动平均': hum_ma,
    '湿度_指数平滑': hum_exp,
    '温度_ZScore': temp_zscore,
    '湿度_ZScore': hum_zscore
})

result_df.to_excel(f'{output_dir}\\DHT11预处理结果数据.xlsx', index=False)
print(f"  数据已保存: {output_dir}\\DHT11预处理结果数据.xlsx")

# ==================== 9. 生成报告 ====================
print("\n【步骤9】生成处理报告...")

# 重新计算相关系数用于报告
if len(temp_clean_valid) > 1:
    correlation_report = np.corrcoef(temp_clean_valid, hum_clean_valid)[0, 1]
else:
    correlation_report = np.nan

report = f"""
{'=' * 70}
DHT11温湿度传感器数据预处理报告
{'=' * 70}

1. 数据概况
   - 采集样本数: {n_samples}
   - 采集时长: 5分钟
   - 采样频率: 1Hz

2. 原始数据统计
   温度:
   - 最小值: {temperature.min():.2f}°C
   - 最大值: {temperature.max():.2f}°C
   - 平均值: {temperature.mean():.2f}°C
   - 标准差: {temperature.std():.2f}°C

   湿度:
   - 最小值: {humidity.min():.2f}%
   - 最大值: {humidity.max():.2f}%
   - 平均值: {humidity.mean():.2f}%
   - 标准差: {humidity.std():.2f}%

3. 异常值检测
   - 温度异常值: {np.sum(temp_outliers_z)}个 (Z-Score方法)
   - 湿度异常值: {np.sum(hum_outliers_z)}个 (Z-Score方法)

4. 滤波效果
   - 温度SG滤波后标准差: {np.nanstd(temp_sg):.4f}°C
   - 湿度指数平滑后标准差: {np.nanstd(hum_exp):.4f}%

5. 相关性分析
   - 温湿度相关系数: {correlation_report:.4f}

6. 预处理算法清单
   ✓ Z-Score异常值检测
   ✓ IQR异常值检测
   ✓ 线性插值
   ✓ 移动平均滤波
   ✓ 指数平滑滤波
   ✓ Savitzky-Golay滤波
   ✓ Z-Score标准化
   ✓ Min-Max归一化
   ✓ Robust标准化

{'=' * 70}
"""

with open(f'{output_dir}\\DHT11预处理报告.txt', 'w', encoding='utf-8') as f:
    f.write(report)

print(report)
print(f"报告已保存: {output_dir}\\DHT11预处理报告.txt")

print("\n" + "=" * 70)
print("DHT11预处理演示完成！")
print(f"输出目录: {output_dir}")
print("=" * 70)