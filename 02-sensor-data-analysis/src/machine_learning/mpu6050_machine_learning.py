#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MPU-6050陀螺仪传感器 - 机器学习完整流程
包含：特征工程、模型训练、融合决策、可视化
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor, StackingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression
import warnings

warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 80)
print("MPU-6050陀螺仪传感器 - 机器学习完整演示")
print("=" * 80)

import os

output_dir = r'D:\高校专项\实验项目\实验三\MPU-6050机器学习'
os.makedirs(output_dir, exist_ok=True)

# ==================== 1. 数据加载 ====================
print("\n【步骤1】数据加载...")

data_path = r'D:\高校专项\实验项目\实验三\MPU-6050机器学习\MPU-6050预处理结果数据.xlsx'
if os.path.exists(data_path):
    df = pd.read_excel(data_path)
    print(f"  加载预处理数据: {len(df)} 条记录")
else:
    print("  生成模拟数据...")
    np.random.seed(46)
    n_samples = 600
    t = np.linspace(0, 4 * np.pi, n_samples)

    # 生成姿态角
    pitch_true = 15 * np.sin(t) + 5 * np.sin(3 * t)
    roll_true = 20 * np.cos(t) * np.sin(t)
    yaw_true = 30 * np.sin(0.5 * t)

    # 生成传感器数据
    ax = -np.sin(np.radians(roll_true)) * np.cos(np.radians(pitch_true)) + np.random.normal(0, 0.02, n_samples)
    ay = np.sin(np.radians(pitch_true)) + np.random.normal(0, 0.02, n_samples)
    az = np.cos(np.radians(roll_true)) * np.cos(np.radians(pitch_true)) + np.random.normal(0, 0.02, n_samples)

    gx = np.gradient(pitch_true, 0.05) + np.random.normal(0, 0.5, n_samples)
    gy = np.gradient(roll_true, 0.05) + np.random.normal(0, 0.5, n_samples)
    gz = np.gradient(yaw_true, 0.05) + np.random.normal(0, 0.5, n_samples)

    df = pd.DataFrame({
        '时间': np.arange(n_samples) * 0.05,
        'Ax': ax, 'Ay': ay, 'Az': az,
        'Gx': gx, 'Gy': gy, 'Gz': gz,
        '真实俯仰角': pitch_true,
        '真实横滚角': roll_true
    })
    print(f"  生成模拟数据: {n_samples} 条记录")

# ==================== 2. 特征工程 ====================
print("\n【步骤2】特征工程...")


class FeatureEngineering:
    @staticmethod
    def create_sensor_features(df):
        """传感器融合特征"""
        # 加速度模值
        df['加速度模'] = np.sqrt(df['Ax'] ** 2 + df['Ay'] ** 2 + df['Az'] ** 2)

        # 角速度模值
        df['角速度模'] = np.sqrt(df['Gx'] ** 2 + df['Gy'] ** 2 + df['Gz'] ** 2)

        # 姿态角估计（加速度计）
        df['俯仰角_估计'] = np.degrees(np.arctan2(df['Ay'], np.sqrt(df['Ax'] ** 2 + df['Az'] ** 2)))
        df['横滚角_估计'] = np.degrees(np.arctan2(-df['Ax'], df['Az']))

        return df

    @staticmethod
    def create_temporal_features(df):
        """时间特征"""
        df['时间_sin'] = np.sin(2 * np.pi * df['时间'] / 30)
        df['时间_cos'] = np.cos(2 * np.pi * df['时间'] / 30)
        return df

    @staticmethod
    def create_statistical_features(df, columns, window=20):
        """统计特征"""
        for col in columns:
            df[f'{col}_mean'] = df[col].rolling(window=window, min_periods=1).mean()
            df[f'{col}_std'] = df[col].rolling(window=window, min_periods=1).std()
            df[f'{col}_max'] = df[col].rolling(window=window, min_periods=1).max()
            df[f'{col}_min'] = df[col].rolling(window=window, min_periods=1).min()
        return df

    @staticmethod
    def create_differential_features(df, columns):
        """微分特征（修复NaN问题）"""
        for col in columns:
            df[f'{col}_diff'] = df[col].diff().fillna(0)
            # 二阶差分：对一阶差分再差分，前两个值为NaN，填充0
            df[f'{col}_diff2'] = df[col].diff().diff().fillna(0)
        return df


# 应用特征工程
df = FeatureEngineering.create_sensor_features(df)
df = FeatureEngineering.create_temporal_features(df)
sensor_cols = ['Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz']
df = FeatureEngineering.create_statistical_features(df, sensor_cols, window=20)
df = FeatureEngineering.create_differential_features(df, sensor_cols)

# ==================== 关键修复：清理NaN ====================
print(f"\n  特征工程后总行数: {len(df)}")
nan_count = df.isnull().sum().sum()
nan_rows = df.isnull().any(axis=1).sum()
if nan_count > 0:
    print(f"  发现 NaN: 总数={nan_count}, 含NaN的行数={nan_rows}")
    # 删除所有包含NaN的行（差分特征导致的前几行NaN）
    df = df.dropna().reset_index(drop=True)
    print(f"  清理后行数: {len(df)} (删除了 {nan_rows} 行)")
else:
    print(f"  数据无缺失值，继续...")

print(f"  特征工程完成，总特征数: {df.shape[1]}")
# ========================================================

# ==================== 3. 多目标预测 ====================
print("\n【步骤3】多目标预测（俯仰角和横滚角）...")

# 特征和目标
feature_cols = [col for col in df.columns if col not in ['时间', '真实俯仰角', '真实横滚角']]
X = df[feature_cols]
y_pitch = df['真实俯仰角']
y_roll = df['真实横滚角']

print(f"  特征数: {len(feature_cols)}")

# 分割数据
X_train, X_test, y_pitch_train, y_pitch_test = train_test_split(X, y_pitch, test_size=0.2, random_state=42)
_, _, y_roll_train, y_roll_test = train_test_split(X, y_roll, test_size=0.2, random_state=42)

# 标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==================== 4. 模型训练（俯仰角） ====================
print("\n【步骤4】训练俯仰角预测模型...")

models = {
    '随机森林': RandomForestRegressor(n_estimators=100, random_state=42),
    '梯度提升': GradientBoostingRegressor(n_estimators=100, random_state=42),
    'SVR': SVR(kernel='rbf', C=1.0),
    '神经网络': MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=2000, random_state=42)
}

pitch_results = {}

for name, model in models.items():
    print(f"  训练: {name}")
    model.fit(X_train_scaled, y_pitch_train)

    y_pred = model.predict(X_test_scaled)
    rmse = np.sqrt(mean_squared_error(y_pitch_test, y_pred))
    r2 = r2_score(y_pitch_test, y_pred)

    pitch_results[name] = {'model': model, 'rmse': rmse, 'r2': r2, 'y_pred': y_pred}
    print(f"    R²: {r2:.4f}, RMSE: {rmse:.4f}°")

# ==================== 5. 模型训练（横滚角） ====================
print("\n【步骤5】训练横滚角预测模型...")

roll_results = {}

for name, model in models.items():
    print(f"  训练: {name}")
    model.fit(X_train_scaled, y_roll_train)

    y_pred = model.predict(X_test_scaled)
    rmse = np.sqrt(mean_squared_error(y_roll_test, y_pred))
    r2 = r2_score(y_roll_test, y_pred)

    roll_results[name] = {'model': model, 'rmse': rmse, 'r2': r2, 'y_pred': y_pred}
    print(f"    R²: {r2:.4f}, RMSE: {rmse:.4f}°")

# ==================== 6. 融合决策 ====================
print("\n【步骤6】融合决策...")

# 俯仰角融合
best_pitch = max(pitch_results.items(), key=lambda x: x[1]['r2'])[0]
pitch_models = sorted(pitch_results.items(), key=lambda x: x[1]['r2'], reverse=True)[:2]
pitch_estimators = [(name, res['model']) for name, res in pitch_models]
pitch_voting = VotingRegressor(pitch_estimators)
pitch_voting.fit(X_train_scaled, y_pitch_train)
y_pitch_voting = pitch_voting.predict(X_test_scaled)

print(f"\n  俯仰角融合:")
print(f"    最佳单模型: {best_pitch} (R²={pitch_results[best_pitch]['r2']:.4f})")
print(f"    Voting融合 R²: {r2_score(y_pitch_test, y_pitch_voting):.4f}")

# 横滚角融合
best_roll = max(roll_results.items(), key=lambda x: x[1]['r2'])[0]
roll_models = sorted(roll_results.items(), key=lambda x: x[1]['r2'], reverse=True)[:2]
roll_estimators = [(name, res['model']) for name, res in roll_models]
roll_voting = VotingRegressor(roll_estimators)
roll_voting.fit(X_train_scaled, y_roll_train)
y_roll_voting = roll_voting.predict(X_test_scaled)

print(f"\n  横滚角融合:")
print(f"    最佳单模型: {best_roll} (R²={roll_results[best_roll]['r2']:.4f})")
print(f"    Voting融合 R²: {r2_score(y_roll_test, y_roll_voting):.4f}")

# ==================== 7. 可视化 ====================
print("\n【步骤7】生成可视化...")

fig = plt.figure(figsize=(16, 12))

# 子图1: 俯仰角模型对比
ax1 = plt.subplot(3, 3, 1)
names = list(pitch_results.keys())
r2_scores = [pitch_results[name]['r2'] for name in names]
ax1.barh(names, r2_scores, color='lightblue')
ax1.set_xlabel('R² Score')
ax1.set_title('俯仰角预测 - 模型对比', fontweight='bold')
for i, v in enumerate(r2_scores):
    ax1.text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=9)

# 子图2: 横滚角模型对比
ax2 = plt.subplot(3, 3, 2)
r2_scores = [roll_results[name]['r2'] for name in names]
ax2.barh(names, r2_scores, color='lightcoral')
ax2.set_xlabel('R² Score')
ax2.set_title('横滚角预测 - 模型对比', fontweight='bold')
for i, v in enumerate(r2_scores):
    ax2.text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=9)

# 子图3: 俯仰角预测
ax3 = plt.subplot(3, 3, 3)
y_pred_best_pitch = pitch_results[best_pitch]['y_pred']
ax3.scatter(y_pitch_test, y_pred_best_pitch, alpha=0.6, edgecolors='black', linewidth=0.5)
ax3.plot([y_pitch_test.min(), y_pitch_test.max()], [y_pitch_test.min(), y_pitch_test.max()], 'r--', lw=2)
ax3.set_xlabel('实际俯仰角 (°)')
ax3.set_ylabel('预测俯仰角 (°)')
ax3.set_title(f'{best_pitch} - 俯仰角预测', fontweight='bold')
ax3.grid(True, alpha=0.3)

# 子图4: 横滚角预测
ax4 = plt.subplot(3, 3, 4)
y_pred_best_roll = roll_results[best_roll]['y_pred']
ax4.scatter(y_roll_test, y_pred_best_roll, alpha=0.6, edgecolors='black', linewidth=0.5)
ax4.plot([y_roll_test.min(), y_roll_test.max()], [y_roll_test.min(), y_roll_test.max()], 'r--', lw=2)
ax4.set_xlabel('实际横滚角 (°)')
ax4.set_ylabel('预测横滚角 (°)')
ax4.set_title(f'{best_roll} - 横滚角预测', fontweight='bold')
ax4.grid(True, alpha=0.3)

# 子图5: 时间序列 - 俯仰角
ax5 = plt.subplot(3, 3, 5)
test_idx = range(len(y_pitch_test))
ax5.plot(test_idx, y_pitch_test, 'b-', label='实际值', linewidth=2)
ax5.plot(test_idx, y_pred_best_pitch, 'r--', label='预测值', linewidth=2)
ax5.set_xlabel('样本索引')
ax5.set_ylabel('俯仰角 (°)')
ax5.set_title('俯仰角时间序列', fontweight='bold')
ax5.legend()
ax5.grid(True, alpha=0.3)

# 子图6: 时间序列 - 横滚角
ax6 = plt.subplot(3, 3, 6)
ax6.plot(test_idx, y_roll_test, 'b-', label='实际值', linewidth=2)
ax6.plot(test_idx, y_pred_best_roll, 'r--', label='预测值', linewidth=2)
ax6.set_xlabel('样本索引')
ax6.set_ylabel('横滚角 (°)')
ax6.set_title('横滚角时间序列', fontweight='bold')
ax6.legend()
ax6.grid(True, alpha=0.3)

# 子图7: 残差分析 - 俯仰角
ax7 = plt.subplot(3, 3, 7)
residuals_pitch = y_pitch_test - y_pred_best_pitch
ax7.hist(residuals_pitch, bins=20, color='skyblue', edgecolor='black')
ax7.axvline(x=0, color='r', linestyle='--')
ax7.set_xlabel('残差 (°)')
ax7.set_ylabel('频数')
ax7.set_title('俯仰角残差分布', fontweight='bold')

# 子图8: 残差分析 - 横滚角
ax8 = plt.subplot(3, 3, 8)
residuals_roll = y_roll_test - y_pred_best_roll
ax8.hist(residuals_roll, bins=20, color='lightgreen', edgecolor='black')
ax8.axvline(x=0, color='r', linestyle='--')
ax8.set_xlabel('残差 (°)')
ax8.set_ylabel('频数')
ax8.set_title('横滚角残差分布', fontweight='bold')

# 子图9: 总结
ax9 = plt.subplot(3, 3, 9)
ax9.axis('off')
summary = f"""
MPU-6050机器学习总结

特征工程:
  • 传感器融合特征
  • 时间编码特征
  • 统计特征(mean/std)
  • 微分特征

俯仰角预测:
  最佳模型: {best_pitch}
  R²={pitch_results[best_pitch]['r2']:.4f}
  RMSE={pitch_results[best_pitch]['rmse']:.4f}°

横滚角预测:
  最佳模型: {best_roll}
  R²={roll_results[best_roll]['r2']:.4f}
  RMSE={roll_results[best_roll]['rmse']:.4f}°

融合效果:
  俯仰角融合 R²={r2_score(y_pitch_test, y_pitch_voting):.4f}
  横滚角融合 R²={r2_score(y_roll_test, y_roll_voting):.4f}
"""
ax9.text(0.1, 0.95, summary, transform=ax9.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.suptitle('MPU-6050陀螺仪传感器 - 机器学习完整流程', fontsize=16, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.96])

plt.savefig(f'{output_dir}\\MPU-6050机器学习结果.png', dpi=150, bbox_inches='tight')
plt.savefig(f'{output_dir}\\MPU-6050机器学习结果_高清.png', dpi=300, bbox_inches='tight')
plt.show()

# ==================== 8. 保存结果 ====================
print("\n【步骤8】保存结果...")

results_df = pd.DataFrame({
    '模型': list(pitch_results.keys()),
    '俯仰角_R2': [pitch_results[name]['r2'] for name in pitch_results.keys()],
    '俯仰角_RMSE': [pitch_results[name]['rmse'] for name in pitch_results.keys()],
    '横滚角_R2': [roll_results[name]['r2'] for name in roll_results.keys()],
    '横滚角_RMSE': [roll_results[name]['rmse'] for name in roll_results.keys()]
})
results_df.to_excel(f'{output_dir}\\模型性能对比.xlsx', index=False)

predictions_df = pd.DataFrame({
    '俯仰角_实际': y_pitch_test.values if hasattr(y_pitch_test, 'values') else y_pitch_test,
    '俯仰角_预测': y_pred_best_pitch,
    '横滚角_实际': y_roll_test.values if hasattr(y_roll_test, 'values') else y_roll_test,
    '横滚角_预测': y_pred_best_roll
})
predictions_df.to_excel(f'{output_dir}\\预测结果.xlsx', index=False)

print(f"  结果已保存到: {output_dir}")

print("\n" + "=" * 80)
print("MPU-6050机器学习演示完成！")
print("=" * 80)