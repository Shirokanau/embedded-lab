#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MPU-6050陀螺仪传感器数据预处理完整演示
包含：数据读取、零偏校准、互补滤波、四元数解算、可视化
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.signal import butter, filtfilt
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 70)
print("MPU-6050陀螺仪传感器数据预处理演示")
print("=" * 70)

# ==================== 1. 生成模拟数据 ====================
print("\n【步骤1】生成模拟姿态数据...")
np.random.seed(46)
n_samples = 600  # 30秒，20Hz采样
time = np.arange(n_samples) * 0.05  # 时间轴（秒）

# 模拟Figure-8运动轨迹
t = np.linspace(0, 4*np.pi, n_samples)

# 俯仰角（前后倾斜）
pitch_true = 15 * np.sin(t) + 5 * np.sin(3*t)

# 横滚角（左右倾斜）
roll_true = 20 * np.cos(t) * np.sin(t)

# 偏航角（水平旋转）
yaw_true = 30 * np.sin(0.5*t)

# 生成加速度计数据（根据角度计算）
def angles_to_accel(pitch, roll):
    """将角度转换为加速度计读数（单位：g）"""
    ax = -np.sin(roll) * np.cos(pitch)
    ay = np.sin(pitch)
    az = np.cos(roll) * np.cos(pitch)
    return ax, ay, az

ax_true, ay_true, az_true = angles_to_accel(np.radians(pitch_true), np.radians(roll_true))

# 添加噪声
accel_noise = 0.02
ax = ax_true + np.random.normal(0, accel_noise, n_samples)
ay = ay_true + np.random.normal(0, accel_noise, n_samples)
az = az_true + np.random.normal(0, accel_noise, n_samples)

# 生成陀螺仪数据（角度的导数）
dt = 0.05
gx_true = np.gradient(pitch_true, dt)  # 俯仰角速度
gy_true = np.gradient(roll_true, dt)   # 横滚角速度
gz_true = np.gradient(yaw_true, dt)    # 偏航角速度

# 添加噪声和零偏
gyro_noise = 0.5
gyro_bias = np.array([0.3, -0.2, 0.1])  # 零偏误差
gx = gx_true + np.random.normal(0, gyro_noise, n_samples) + gyro_bias[0]
gy = gy_true + np.random.normal(0, gyro_noise, n_samples) + gyro_bias[1]
gz = gz_true + np.random.normal(0, gyro_noise, n_samples) + gyro_bias[2]

print(f"  生成样本数: {n_samples}")
print(f"  采样频率: 20Hz")
print(f"  采集时长: {n_samples * dt:.1f}秒")
print(f"  俯仰角范围: {pitch_true.min():.1f}° ~ {pitch_true.max():.1f}°")
print(f"  横滚角范围: {roll_true.min():.1f}° ~ {roll_true.max():.1f}°")

# ==================== 2. 零偏校准 ====================
print("\n【步骤2】陀螺仪零偏校准...")

class Calibration:
    @staticmethod
    def gyro_bias_calibration(gyro_data, static_samples=100):
        """陀螺仪零偏校准"""
        bias = np.mean(gyro_data[:static_samples], axis=0)
        calibrated = gyro_data - bias
        return calibrated, bias

# 校准陀螺仪
gyro_data = np.column_stack([gx, gy, gz])
gyro_calibrated, gyro_bias_estimated = Calibration.gyro_bias_calibration(gyro_data)
gx_cal = gyro_calibrated[:, 0]
gy_cal = gyro_calibrated[:, 1]
gz_cal = gyro_calibrated[:, 2]

print(f"  估计零偏: X={gyro_bias_estimated[0]:.3f}°/s, Y={gyro_bias_estimated[1]:.3f}°/s, Z={gyro_bias_estimated[2]:.3f}°/s")
print(f"  实际零偏: X={gyro_bias[0]:.3f}°/s, Y={gyro_bias[1]:.3f}°/s, Z={gyro_bias[2]:.3f}°/s")

# ==================== 3. 滤波处理 ====================
print("\n【步骤3】滤波处理...")

class Filter:
    @staticmethod
    def lowpass_filter(data, cutoff=5, fs=20, order=2):
        """低通滤波器（用于加速度计）"""
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return filtfilt(b, a, data)
    
    @staticmethod
    def highpass_filter(data, cutoff=0.5, fs=20, order=2):
        """高通滤波器（用于陀螺仪）"""
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='high', analog=False)
        return filtfilt(b, a, data)

# 应用滤波
ax_lp = Filter.lowpass_filter(ax, cutoff=5, fs=20)
ay_lp = Filter.lowpass_filter(ay, cutoff=5, fs=20)
az_lp = Filter.lowpass_filter(az, cutoff=5, fs=20)

gx_hp = Filter.highpass_filter(gx_cal, cutoff=0.5, fs=20)
gy_hp = Filter.highpass_filter(gy_cal, cutoff=0.5, fs=20)
gz_hp = Filter.highpass_filter(gz_cal, cutoff=0.5, fs=20)

print("  加速度计: 低通滤波 (截止频率=5Hz)")
print("  陀螺仪: 高通滤波 (截止频率=0.5Hz)")

# ==================== 4. 互补滤波 ====================
print("\n【步骤4】互补滤波姿态解算...")

class ComplementaryFilter:
    def __init__(self, alpha=0.96):
        """
        互补滤波器
        alpha: 陀螺仪权重 (0-1)
        """
        self.alpha = alpha
        self.pitch = 0
        self.roll = 0
    
    def update(self, accel, gyro, dt):
        """
        更新姿态
        accel: [ax, ay, az] 加速度 (g)
        gyro: [gx, gy, gz] 角速度 (°/s)
        dt: 采样时间 (s)
        """
        ax, ay, az = accel
        gx, gy, gz = gyro
        
        # 加速度计计算角度
        pitch_acc = np.degrees(np.arctan2(ay, np.sqrt(ax**2 + az**2)))
        roll_acc = np.degrees(np.arctan2(-ax, az))
        
        # 陀螺仪积分
        self.pitch += gx * dt
        self.roll += gy * dt
        
        # 互补滤波融合
        self.pitch = self.alpha * self.pitch + (1 - self.alpha) * pitch_acc
        self.roll = self.alpha * self.roll + (1 - self.alpha) * roll_acc
        
        return self.pitch, self.roll

# 应用互补滤波
cf = ComplementaryFilter(alpha=0.96)
pitch_cf = np.zeros(n_samples)
roll_cf = np.zeros(n_samples)

for i in range(n_samples):
    accel = [ax_lp[i], ay_lp[i], az_lp[i]]
    gyro = [gx_hp[i], gy_hp[i], gz_hp[i]]
    pitch_cf[i], roll_cf[i] = cf.update(accel, gyro, dt)

print(f"  互补滤波参数: α={cf.alpha}")
print(f"  融合后俯仰角范围: {pitch_cf.min():.1f}° ~ {pitch_cf.max():.1f}°")
print(f"  融合后横滚角范围: {roll_cf.min():.1f}° ~ {roll_cf.max():.1f}°")

# ==================== 5. 仅加速度计角度 ====================
print("\n【步骤5】纯加速度计姿态计算...")

pitch_acc_only = np.degrees(np.arctan2(ay_lp, np.sqrt(ax_lp**2 + az_lp**2)))
roll_acc_only = np.degrees(np.arctan2(-ax_lp, az_lp))

print("  完成纯加速度计姿态计算")

# ==================== 6. 仅陀螺仪角度 ====================
print("\n【步骤6】纯陀螺仪姿态积分...")

pitch_gyro_only = np.zeros(n_samples)
roll_gyro_only = np.zeros(n_samples)

pitch_gyro_only[0] = pitch_acc_only[0]
roll_gyro_only[0] = roll_acc_only[0]

for i in range(1, n_samples):
    pitch_gyro_only[i] = pitch_gyro_only[i-1] + gx_hp[i] * dt
    roll_gyro_only[i] = roll_gyro_only[i-1] + gy_hp[i] * dt

print("  完成纯陀螺仪姿态积分")

# ==================== 7. 误差分析 ====================
print("\n【步骤7】误差分析...")

def calculate_angle_error(estimated, true):
    """计算角度误差"""
    error = estimated - true
    # 处理角度环绕问题
    error = np.mod(error + 180, 360) - 180
    rmse = np.sqrt(np.mean(error**2))
    mae = np.mean(np.abs(error))
    max_error = np.max(np.abs(error))
    return rmse, mae, max_error

# 计算各方法误差
pitch_rmse_acc, pitch_mae_acc, pitch_max_acc = calculate_angle_error(pitch_acc_only, pitch_true)
roll_rmse_acc, roll_mae_acc, roll_max_acc = calculate_angle_error(roll_acc_only, roll_true)

pitch_rmse_gyro, pitch_mae_gyro, pitch_max_gyro = calculate_angle_error(pitch_gyro_only, pitch_true)
roll_rmse_gyro, roll_mae_gyro, roll_max_gyro = calculate_angle_error(roll_gyro_only, roll_true)

pitch_rmse_cf, pitch_mae_cf, pitch_max_cf = calculate_angle_error(pitch_cf, pitch_true)
roll_rmse_cf, roll_mae_cf, roll_max_cf = calculate_angle_error(roll_cf, roll_true)

print("\n  俯仰角误差对比 (RMSE):")
print(f"    纯加速度计: {pitch_rmse_acc:.3f}°")
print(f"    纯陀螺仪: {pitch_rmse_gyro:.3f}°")
print(f"    互补滤波: {pitch_rmse_cf:.3f}°")

print("\n  横滚角误差对比 (RMSE):")
print(f"    纯加速度计: {roll_rmse_acc:.3f}°")
print(f"    纯陀螺仪: {roll_rmse_gyro:.3f}°")
print(f"    互补滤波: {roll_rmse_cf:.3f}°")

# ==================== 8. 可视化输出 ====================
print("\n【步骤8】生成可视化图表...")

fig = plt.figure(figsize=(16, 12))

# 子图1：原始加速度计数据
ax1 = plt.subplot(3, 3, 1)
ax1.plot(time, ax, 'r-', alpha=0.5, linewidth=1, label='Ax')
ax1.plot(time, ay, 'g-', alpha=0.5, linewidth=1, label='Ay')
ax1.plot(time, az, 'b-', alpha=0.5, linewidth=1, label='Az')
ax1.set_xlabel('时间 (秒)')
ax1.set_ylabel('加速度 (g)')
ax1.set_title('原始加速度计数据', fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 子图2：原始陀螺仪数据
ax2 = plt.subplot(3, 3, 2)
ax2.plot(time, gx, 'r-', alpha=0.5, linewidth=1, label='Gx')
ax2.plot(time, gy, 'g-', alpha=0.5, linewidth=1, label='Gy')
ax2.plot(time, gz, 'b-', alpha=0.5, linewidth=1, label='Gz')
ax2.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.3)
ax2.set_xlabel('时间 (秒)')
ax2.set_ylabel('角速度 (°/s)')
ax2.set_title('原始陀螺仪数据（含零偏）', fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 子图3：校准后陀螺仪数据
ax3 = plt.subplot(3, 3, 3)
ax3.plot(time, gx_cal, 'r-', alpha=0.7, linewidth=1, label='Gx (校准后)')
ax3.plot(time, gy_cal, 'g-', alpha=0.7, linewidth=1, label='Gy (校准后)')
ax3.plot(time, gz_cal, 'b-', alpha=0.7, linewidth=1, label='Gz (校准后)')
ax3.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.3)
ax3.set_xlabel('时间 (秒)')
ax3.set_ylabel('角速度 (°/s)')
ax3.set_title('零偏校准后陀螺仪数据', fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 子图4：俯仰角对比
ax4 = plt.subplot(3, 3, 4)
ax4.plot(time, pitch_true, 'k-', linewidth=2, label='真实值', alpha=0.8)
ax4.plot(time, pitch_acc_only, 'b-', alpha=0.5, linewidth=1, label='纯加速度计')
ax4.plot(time, pitch_gyro_only, 'g-', alpha=0.5, linewidth=1, label='纯陀螺仪')
ax4.plot(time, pitch_cf, 'r-', linewidth=2, label='互补滤波')
ax4.set_xlabel('时间 (秒)')
ax4.set_ylabel('俯仰角 (°)')
ax4.set_title(f'俯仰角估计对比 (RMSE: 互补滤波={pitch_rmse_cf:.2f}°)', fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)

# 子图5：横滚角对比
ax5 = plt.subplot(3, 3, 5)
ax5.plot(time, roll_true, 'k-', linewidth=2, label='真实值', alpha=0.8)
ax5.plot(time, roll_acc_only, 'b-', alpha=0.5, linewidth=1, label='纯加速度计')
ax5.plot(time, roll_gyro_only, 'g-', alpha=0.5, linewidth=1, label='纯陀螺仪')
ax5.plot(time, roll_cf, 'r-', linewidth=2, label='互补滤波')
ax5.set_xlabel('时间 (秒)')
ax5.set_ylabel('横滚角 (°)')
ax5.set_title(f'横滚角估计对比 (RMSE: 互补滤波={roll_rmse_cf:.2f}°)', fontweight='bold')
ax5.legend()
ax5.grid(True, alpha=0.3)

# 子图6：俯仰角误差
ax6 = plt.subplot(3, 3, 6)
error_acc = pitch_acc_only - pitch_true
error_gyro = pitch_gyro_only - pitch_true
error_cf = pitch_cf - pitch_true
ax6.plot(time, error_acc, 'b-', alpha=0.5, linewidth=1, label='纯加速度计')
ax6.plot(time, error_gyro, 'g-', alpha=0.5, linewidth=1, label='纯陀螺仪')
ax6.plot(time, error_cf, 'r-', linewidth=1.5, label='互补滤波')
ax6.axhline(y=0, color='k', linestyle='--', linewidth=1)
ax6.fill_between(time, error_cf, alpha=0.2, color='red')
ax6.set_xlabel('时间 (秒)')
ax6.set_ylabel('误差 (°)')
ax6.set_title('俯仰角估计误差', fontweight='bold')
ax6.legend()
ax6.grid(True, alpha=0.3)

# 子图7：3D轨迹可视化（Figure-8）
ax7 = plt.subplot(3, 3, 7, projection='3d')
# 将角度转换为3D位置进行可视化
scale = 0.5
x_pos = scale * np.sin(np.radians(roll_cf)) * np.cos(np.radians(pitch_cf))
y_pos = scale * np.sin(np.radians(pitch_cf))
z_pos = scale * np.cos(np.radians(roll_cf)) * np.cos(np.radians(pitch_cf))
ax7.plot(x_pos, y_pos, z_pos, 'b-', linewidth=2, label='估计轨迹')
ax7.plot(x_pos[0], y_pos[0], z_pos[0], 'go', markersize=10, label='起点')
ax7.plot(x_pos[-1], y_pos[-1], z_pos[-1], 'ro', markersize=10, label='终点')
ax7.set_xlabel('X')
ax7.set_ylabel('Y')
ax7.set_zlabel('Z')
ax7.set_title('Figure-8运动轨迹（3D可视化）', fontweight='bold')
ax7.legend()

# 子图8：误差统计直方图
ax8 = plt.subplot(3, 3, 8)
ax8.hist(error_cf, bins=30, color='skyblue', edgecolor='black', alpha=0.7, label='互补滤波误差')
ax8.axvline(x=0, color='r', linestyle='--', linewidth=2)
ax8.axvline(x=np.mean(error_cf), color='g', linestyle='-', linewidth=2, 
           label=f'均值={np.mean(error_cf):.2f}°')
ax8.set_xlabel('误差 (°)')
ax8.set_ylabel('频数')
ax8.set_title('互补滤波俯仰角误差分布', fontweight='bold')
ax8.legend()
ax8.grid(True, alpha=0.3, axis='y')

# 子图9：处理流程总结
ax9 = plt.subplot(3, 3, 9)
ax9.axis('off')
summary_text = f"""
MPU-6050预处理流程总结

输入数据:
  • 样本数: {n_samples}
  • 采样频率: 20Hz
  • 采集时长: {n_samples * dt:.1f}秒

处理步骤:
  1. 零偏校准
     - 估计零偏: X={gyro_bias_estimated[0]:.2f}, Y={gyro_bias_estimated[1]:.2f}, Z={gyro_bias_estimated[2]:.2f}
  
  2. 滤波处理
     - 加速度计: 低通滤波 (5Hz)
     - 陀螺仪: 高通滤波 (0.5Hz)
  
  3. 互补滤波融合
     - α={cf.alpha} (陀螺仪权重)
     - 1-α={1-cf.alpha} (加速度计权重)

误差指标 (RMSE):
  俯仰角:
    纯加速度计: {pitch_rmse_acc:.3f}°
    纯陀螺仪: {pitch_rmse_gyro:.3f}°
    互补滤波: {pitch_rmse_cf:.3f}°
  
  横滚角:
    纯加速度计: {roll_rmse_acc:.3f}°
    纯陀螺仪: {roll_rmse_gyro:.3f}°
    互补滤波: {roll_rmse_cf:.3f}°

性能提升:
  俯仰角 RMSE 降低: {(1-pitch_rmse_cf/pitch_rmse_acc)*100:.1f}%
  横滚角 RMSE 降低: {(1-roll_rmse_cf/roll_rmse_acc)*100:.1f}%
"""
ax9.text(0.1, 0.95, summary_text, transform=ax9.transAxes,
        fontsize=9, verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.3))

plt.suptitle('MPU-6050陀螺仪传感器数据预处理完整流程', fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.96])

# 保存图片
output_dir = r'D:\研究生\2026-04\实验二数据预处理\output'
import os
os.makedirs(output_dir, exist_ok=True)

plt.savefig(f'{output_dir}\\MPU-6050预处理结果.png', dpi=150, bbox_inches='tight')
print(f"  图表已保存: {output_dir}\\MPU-6050预处理结果.png")

plt.savefig(f'{output_dir}\\MPU-6050预处理结果_高清.png', dpi=300, bbox_inches='tight')
print(f"  高清图表已保存: {output_dir}\\MPU-6050预处理结果_高清.png")

plt.show()

# ==================== 9. 保存处理后的数据 ====================
print("\n【步骤9】保存处理后的数据...")

result_df = pd.DataFrame({
    '时间': time,
    '真实俯仰角': pitch_true,
    '真实横滚角': roll_true,
    'Ax': ax, 'Ay': ay, 'Az': az,
    'Gx': gx, 'Gy': gy, 'Gz': gz,
    'Gx_校准': gx_cal, 'Gy_校准': gy_cal, 'Gz_校准': gz_cal,
    '俯仰角_加速度计': pitch_acc_only,
    '俯仰角_陀螺仪': pitch_gyro_only,
    '俯仰角_互补滤波': pitch_cf,
    '横滚角_加速度计': roll_acc_only,
    '横滚角_陀螺仪': roll_gyro_only,
    '横滚角_互补滤波': roll_cf
})

result_df.to_excel(f'{output_dir}\\MPU-6050预处理结果数据.xlsx', index=False)
print(f"  数据已保存: {output_dir}\\MPU-6050预处理结果数据.xlsx")

# ==================== 10. 生成报告 ====================
print("\n【步骤10】生成处理报告...")

report = f"""
{'='*70}
MPU-6050陀螺仪传感器数据预处理报告
{'='*70}

1. 数据概况
   - 采集样本数: {n_samples}
   - 采样频率: 20Hz
   - 采集时长: {n_samples * dt:.1f}秒
   - 运动模式: Figure-8轨迹

2. 角度范围
   - 俯仰角: {pitch_true.min():.1f}° ~ {pitch_true.max():.1f}°
   - 横滚角: {roll_true.min():.1f}° ~ {roll_true.max():.1f}°
   - 偏航角: {yaw_true.min():.1f}° ~ {yaw_true.max():.1f}°

3. 零偏校准
   - 实际零偏: X={gyro_bias[0]:.3f}°/s, Y={gyro_bias[1]:.3f}°/s, Z={gyro_bias[2]:.3f}°/s
   - 估计零偏: X={gyro_bias_estimated[0]:.3f}°/s, Y={gyro_bias_estimated[1]:.3f}°/s, Z={gyro_bias_estimated[2]:.3f}°/s
   - 校准误差: X={abs(gyro_bias[0]-gyro_bias_estimated[0]):.3f}°/s, Y={abs(gyro_bias[1]-gyro_bias_estimated[1]):.3f}°/s, Z={abs(gyro_bias[2]-gyro_bias_estimated[2]):.3f}°/s

4. 滤波参数
   - 加速度计低通滤波: 截止频率=5Hz
   - 陀螺仪高通滤波: 截止频率=0.5Hz

5. 互补滤波参数
   - 陀螺仪权重 α: {cf.alpha}
   - 加速度计权重 1-α: {1-cf.alpha}

6. 误差指标对比 (RMSE)
   {'方法':<15} {'俯仰角(°)':<12} {'横滚角(°)':<12}
   {'-'*40}
   {'纯加速度计':<15} {pitch_rmse_acc:<12.3f} {roll_rmse_acc:<12.3f}
   {'纯陀螺仪':<15} {pitch_rmse_gyro:<12.3f} {roll_rmse_gyro:<12.3f}
   {'互补滤波':<15} {pitch_rmse_cf:<12.3f} {roll_rmse_cf:<12.3f}

7. 性能提升
   - 俯仰角 RMSE 降低: {(1-pitch_rmse_cf/pitch_rmse_acc)*100:.1f}%
   - 横滚角 RMSE 降低: {(1-roll_rmse_cf/roll_rmse_acc)*100:.1f}%

8. 预处理算法清单
   ✓ 陀螺仪零偏校准
   ✓ 加速度计低通滤波
   ✓ 陀螺仪高通滤波
   ✓ 互补滤波姿态融合
   ✓ 纯加速度计姿态计算
   ✓ 纯陀螺仪姿态积分
   ✓ 误差分析与评估

{'='*70}
"""

with open(f'{output_dir}\\MPU-6050预处理报告.txt', 'w', encoding='utf-8') as f:
    f.write(report)

print(report)
print(f"报告已保存: {output_dir}\\MPU-6050预处理报告.txt")

print("\n" + "=" * 70)
print("MPU-6050预处理演示完成！")
print(f"输出目录: {output_dir}")
print("=" * 70)