#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MPU-6050陀螺仪传感器 - 机器学习完整流程
纯读取外部Excel数据（删除模拟数据）
多目标预测：俯仰角和横滚角
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
import warnings
import os

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 70)
print("MPU-6050 Gyroscope Sensor - Machine Learning Demo")
print("=" * 70)

# ===================== 路径配置 =====================
# Excel数据文件路径（和你生成表格的路径保持一致）
data_path = r'D:\高校专项\实验项目\实验三\MPU-6050机器学习\MPU-6050预处理结果数据.xlsx'
# 结果输出文件夹
output_dir = r'D:\高校专项\实验项目\实验三\MPU-6050机器学习'
os.makedirs(output_dir, exist_ok=True)

# ===================== 1. 读取预处理Excel数据 =====================
print("\n[Step 1] Load preprocessed Excel data...")
# 读取xlsx文件，指定引擎
df = pd.read_excel(data_path, engine="openpyxl")
print(f"Loaded data: {len(df)} rows")
print(f"Data columns: {list(df.columns)}")

# ===================== 2. 特征工程 =====================
print("\n[Step 2] Feature engineering...")

# 传感器融合特征
df['accel_mag'] = np.sqrt(df['Ax']**2 + df['Ay']**2 + df['Az']**2)
df['gyro_mag'] = np.sqrt(df['Gx']**2 + df['Gy']**2 + df['Gz']**2)
df['pitch_est'] = np.degrees(np.arctan2(df['Ay'], np.sqrt(df['Ax']**2 + df['Az']**2)))
df['roll_est'] = np.degrees(np.arctan2(-df['Ax'], df['Az']))

# 时间特征
df['time_sin'] = np.sin(2 * np.pi * df['时间'] / 30)
df['time_cos'] = np.cos(2 * np.pi * df['时间'] / 30)

# 统计特征（滑动窗口）
for col in ['Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz']:
    df[f'{col}_mean'] = df[col].rolling(window=20, min_periods=1).mean()
    df[f'{col}_std'] = df[col].rolling(window=20, min_periods=1).std()

# 微分特征
for col in ['Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz']:
    df[f'{col}_diff'] = df[col].diff().fillna(0)

# 填充特征工程产生的缺失值
df = df.fillna(df.mean())
print(f"Feature count: {df.shape[1]}")

# ===================== 3. 划分特征与标签 =====================
feature_cols = [c for c in df.columns if c not in ['时间', '真实俯仰角', '真实横滚角']]
X = df[feature_cols]
y_pitch = df['真实俯仰角']
y_roll = df['真实横滚角']

# 数据集划分
X_train, X_test, y_pitch_train, y_pitch_test = train_test_split(X, y_pitch, test_size=0.2, random_state=42)
_, _, y_roll_train, y_roll_test = train_test_split(X, y_roll, test_size=0.2, random_state=42)

# 标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Train set: {len(X_train)}, Test set: {len(X_test)}")

# ===================== 4. 训练俯仰角模型 =====================
print("\n[Step 3] Training pitch angle prediction models...")
pitch_models = {
    'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
    'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
    'NeuralNetwork': MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=2000, random_state=42)
}

pitch_results = {}
for name, model in pitch_models.items():
    model.fit(X_train_scaled, y_pitch_train)
    y_pred = model.predict(X_test_scaled)
    rmse = np.sqrt(mean_squared_error(y_pitch_test, y_pred))
    r2 = r2_score(y_pitch_test, y_pred)
    pitch_results[name] = {'rmse': rmse, 'r2': r2, 'y_pred': y_pred}
    print(f"  {name}: R2={r2:.4f}, RMSE={rmse:.4f}deg")

# ===================== 5. 训练横滚角模型 =====================
print("\n[Step 4] Training roll angle prediction models...")
roll_results = {}
for name, model in pitch_models.items():
    model.fit(X_train_scaled, y_roll_train)
    y_pred = model.predict(X_test_scaled)
    rmse = np.sqrt(mean_squared_error(y_roll_test, y_pred))
    r2 = r2_score(y_roll_test, y_pred)
    roll_results[name] = {'rmse': rmse, 'r2': r2, 'y_pred': y_pred}
    print(f"  {name}: R2={r2:.4f}, RMSE={rmse:.4f}deg")

# ===================== 6. 模型融合 =====================
print("\n[Step 5] Fusion decision...")
# 俯仰角融合
best_pitch = max(pitch_results.items(), key=lambda x: x[1]['r2'])[0]
pitch_estimators = [(name, pitch_models[name]) for name in pitch_results.keys()]
pitch_voting = VotingRegressor(pitch_estimators)
pitch_voting.fit(X_train_scaled, y_pitch_train)
y_pitch_voting = pitch_voting.predict(X_test_scaled)

print(f"\n  Pitch angle:")
print(f"    Best single: {best_pitch} (R2={pitch_results[best_pitch]['r2']:.4f})")
print(f"    Voting fusion: R2={r2_score(y_pitch_test, y_pitch_voting):.4f}")

# 横滚角融合
best_roll = max(roll_results.items(), key=lambda x: x[1]['r2'])[0]
roll_estimators = [(name, pitch_models[name]) for name in roll_results.keys()]
roll_voting = VotingRegressor(roll_estimators)
roll_voting.fit(X_train_scaled, y_roll_train)
y_roll_voting = roll_voting.predict(X_test_scaled)

print(f"\n  Roll angle:")
print(f"    Best single: {best_roll} (R2={roll_results[best_roll]['r2']:.4f})")
print(f"    Voting fusion: R2={r2_score(y_roll_test, y_roll_voting):.4f}")

# ===================== 7. 可视化 =====================
print("\n[Step 6] Generating visualization...")
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 俯仰角模型R2对比
ax = axes[0, 0]
names = list(pitch_results.keys())
r2_scores = [pitch_results[n]['r2'] for n in names]
ax.barh(names, r2_scores, color='lightblue')
ax.set_xlabel('R2 Score')
ax.set_title('Pitch Angle - Model Comparison')

# 横滚角模型R2对比
ax = axes[0, 1]
r2_scores = [roll_results[n]['r2'] for n in names]
ax.barh(names, r2_scores, color='lightcoral')
ax.set_xlabel('R2 Score')
ax.set_title('Roll Angle - Model Comparison')

# 俯仰角 真实vs预测
ax = axes[0, 2]
ax.scatter(y_pitch_test, pitch_results[best_pitch]['y_pred'], alpha=0.6)
ax.plot([y_pitch_test.min(), y_pitch_test.max()], [y_pitch_test.min(), y_pitch_test.max()], 'r--')
ax.set_xlabel('Actual Pitch (deg)')
ax.set_ylabel('Predicted Pitch (deg)')
ax.set_title(f'{best_pitch}: Pitch Prediction')

# 横滚角 真实vs预测
ax = axes[1, 0]
ax.scatter(y_roll_test, roll_results[best_roll]['y_pred'], alpha=0.6)
ax.plot([y_roll_test.min(), y_roll_test.max()], [y_roll_test.min(), y_roll_test.max()], 'r--')
ax.set_xlabel('Actual Roll (deg)')
ax.set_ylabel('Predicted Roll (deg)')
ax.set_title(f'{best_roll}: Roll Prediction')

# 俯仰角时序曲线
ax = axes[1, 1]
ax.plot(y_pitch_test.values, label='Actual', linewidth=2)
ax.plot(pitch_results[best_pitch]['y_pred'], label='Predicted', linestyle='--', linewidth=2)
ax.set_xlabel('Sample Index')
ax.set_ylabel('Pitch Angle (deg)')
ax.set_title('Pitch Angle Time Series')
ax.legend()

# 横滚角时序曲线
ax = axes[1, 2]
ax.plot(y_roll_test.values, label='Actual', linewidth=2)
ax.plot(roll_results[best_roll]['y_pred'], label='Predicted', linestyle='--', linewidth=2)
ax.set_xlabel('Sample Index')
ax.set_ylabel('Roll Angle (deg)')
ax.set_title('Roll Angle Time Series')
ax.legend()

plt.suptitle('MPU-6050 Machine Learning - Multi-Target Prediction', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{output_dir}\\MPU6050_ML_Results.png', dpi=150)
plt.show()

# ===================== 8. 保存结果表格 =====================
results_df = pd.DataFrame({
    'Model': list(pitch_results.keys()),
    'Pitch_R2': [pitch_results[n]['r2'] for n in pitch_results.keys()],
    'Pitch_RMSE': [pitch_results[n]['rmse'] for n in pitch_results.keys()],
    'Roll_R2': [roll_results[n]['r2'] for n in roll_results.keys()],
    'Roll_RMSE': [roll_results[n]['rmse'] for n in roll_results.keys()]
})
results_df.to_excel(f'{output_dir}\\Model_Performance.xlsx', index=False, engine="openpyxl")

print(f"\nResults saved to: {output_dir}")
print("\n" + "=" * 70)
print("MPU-6050 Machine Learning Demo Complete!")
print("=" * 70)