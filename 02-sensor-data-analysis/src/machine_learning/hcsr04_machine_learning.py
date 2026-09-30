#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HC-SR04超声波传感器 - 机器学习完整流程
包含：特征工程、模型训练、融合决策、可视化
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 80)
print("HC-SR04超声波传感器 - 机器学习完整演示")
print("=" * 80)

import os
output_dir = r'D:\高校专项\实验项目\实验三\05_HCSR04机器学习'
os.makedirs(output_dir, exist_ok=True)

# ==================== 1. 数据加载 ====================
print("\n【步骤1】数据加载与预处理...")

data_path =  r'D:\高校专项\实验项目\实验三\05_HCSR04机器学习\HC-SR04预处理结果数据.xlsx'
if os.path.exists(data_path):
    df = pd.read_excel(data_path)
    print(f"  加载预处理数据: {len(df)} 条记录")
else:
    print("  生成模拟数据...")
    np.random.seed(44)
    n_samples = 300
    time = np.arange(n_samples)
    
    # 生成距离数据（10cm目标）
    true_distance = 10.0
    noise = np.random.normal(0, 0.5, n_samples)
    drift = 0.02 * np.sin(time / 50)
    outliers = np.zeros(n_samples)
    outlier_idx = np.random.choice(n_samples, 15, replace=False)
    outliers[outlier_idx] = np.random.choice([-1, 1], 15) * np.random.uniform(2, 5, 15)
    distance = true_distance + noise + drift + outliers
    
    # 生成特征
    velocity = np.gradient(distance)
    acceleration = np.gradient(velocity)
    
    df = pd.DataFrame({
        '时间': time,
        '距离': distance,
        '速度': velocity,
        '加速度': acceleration,
        '真实距离': true_distance
    })
    print(f"  生成模拟数据: {n_samples} 条记录")

# ==================== 2. 特征工程 ====================
print("\n【步骤2】特征工程...")

class FeatureEngineering:
    @staticmethod
    def create_temporal_features(df):
        """时间特征"""
        df['时间_sin'] = np.sin(2 * np.pi * df['时间'] / 300)
        df['时间_cos'] = np.cos(2 * np.pi * df['时间'] / 300)
        return df
    
    @staticmethod
    def create_statistical_features(df, column, windows=[5, 10, 20]):
        """统计特征"""
        for window in windows:
            df[f'{column}_mean_{window}'] = df[column].rolling(window=window, min_periods=1).mean()
            df[f'{column}_std_{window}'] = df[column].rolling(window=window, min_periods=1).std()
            df[f'{column}_ema_{window}'] = df[column].ewm(span=window, min_periods=1).mean()
        return df
    
    @staticmethod
    def create_lag_features(df, column, lags=[1, 2, 3, 5, 10]):
        """滞后特征"""
        for lag in lags:
            df[f'{column}_lag{lag}'] = df[column].shift(lag)
        df.fillna(method='bfill', inplace=True)
        return df
    
    @staticmethod
    def create_physics_features(df):
        """物理特征"""
        # 声速温度补偿
        df['声速_估计'] = 331.4 + 0.6 * 20  # 假设20°C
        df['距离_修正'] = df['距离'] * (331.4 / df['声速_估计'])
        
        # 运动状态
        df['运动状态'] = np.where(np.abs(df['速度']) > 0.1, 1, 0)
        df['方向'] = np.sign(df['速度'])
        
        return df

# 应用特征工程
df = FeatureEngineering.create_temporal_features(df)
df = FeatureEngineering.create_statistical_features(df, '距离', windows=[5, 10, 20])
df = FeatureEngineering.create_lag_features(df, '距离', lags=[1, 3, 5])
df = FeatureEngineering.create_physics_features(df)

print(f"  特征工程完成，总特征数: {df.shape[1]}")

# ==================== 3. 特征选择 ====================
print("\n【步骤3】特征选择...")

feature_cols = [col for col in df.columns if col not in ['时间', '真实距离']]
X = df[feature_cols]
y = df['真实距离'] if '真实距离' in df.columns else df['距离']

print(f"  原始特征数: {len(feature_cols)}")

# 使用互信息选择特征
selector = SelectKBest(score_func=mutual_info_regression, k=15)
X_selected = selector.fit_transform(X, y)
selected_features = X.columns[selector.get_support()].tolist()

print(f"  选择后特征数: {len(selected_features)}")
print(f"  选择的特征: {selected_features}")

# ==================== 4. 数据分割 ====================
print("\n【步骤4】数据分割与标准化...")

X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"  训练集: {len(X_train)}, 测试集: {len(X_test)}")

# ==================== 5. 模型训练 ====================
print("\n【步骤5】模型训练...")

models = {
    '线性回归': LinearRegression(),
    '岭回归': Ridge(alpha=1.0),
    'Lasso': Lasso(alpha=0.01),
    'ElasticNet': ElasticNet(alpha=0.01, l1_ratio=0.5),
    '随机森林': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
    '梯度提升': GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
    'SVR': SVR(kernel='rbf', C=1.0, gamma='scale'),
    '神经网络': MLPRegressor(hidden_layer_sizes=(100, 50, 25), max_iter=2000, random_state=42)
}

results = {}

for name, model in models.items():
    print(f"\n  训练: {name}")
    model.fit(X_train_scaled, y_train)
    
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)
    
    results[name] = {
        'model': model,
        'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
        'train_mae': mean_absolute_error(y_train, y_pred_train),
        'test_mae': mean_absolute_error(y_test, y_pred_test),
        'train_r2': r2_score(y_train, y_pred_train),
        'test_r2': r2_score(y_test, y_pred_test),
        'y_pred_test': y_pred_test
    }
    
    print(f"    测试R²: {results[name]['test_r2']:.4f}, RMSE: {results[name]['test_rmse']:.4f}cm")

# ==================== 6. 超参数优化 ====================
print("\n【步骤6】超参数优化（随机森林）...")

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(
    RandomForestRegressor(random_state=42),
    param_grid,
    cv=5,
    scoring='r2',
    n_jobs=-1
)

grid_search.fit(X_train_scaled, y_train)
print(f"  最佳参数: {grid_search.best_params_}")
print(f"  最佳CV R²: {grid_search.best_score_:.4f}")

best_rf = grid_search.best_estimator_
y_pred_best_rf = best_rf.predict(X_test_scaled)
print(f"  优化后测试R²: {r2_score(y_test, y_pred_best_rf):.4f}")

# ==================== 7. 融合决策 ====================
print("\n【步骤7】融合决策...")

# 选择最佳3个模型
best_models = sorted(results.items(), key=lambda x: x[1]['test_r2'], reverse=True)[:3]
print(f"  融合模型: {[name for name, _ in best_models]}")

# Voting Regressor
estimators = [(name, results[name]['model']) for name, _ in best_models]
voting_reg = VotingRegressor(estimators=estimators)
voting_reg.fit(X_train_scaled, y_train)
y_pred_voting = voting_reg.predict(X_test_scaled)

print(f"\n  Voting融合:")
print(f"    R²: {r2_score(y_test, y_pred_voting):.4f}")
print(f"    RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_voting)):.4f}cm")

# 加权融合
weights = np.array([res['test_r2'] for _, res in best_models])
weights = weights / weights.sum()
predictions = np.array([results[name]['y_pred_test'] for name, _ in best_models])
y_pred_weighted = np.average(predictions, axis=0, weights=weights)

print(f"\n  加权融合:")
print(f"    权重: {weights}")
print(f"    R²: {r2_score(y_test, y_pred_weighted):.4f}")
print(f"    RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_weighted)):.4f}cm")

# ==================== 8. 可视化 ====================
print("\n【步骤8】生成可视化...")

fig = plt.figure(figsize=(16, 12))

# 子图1: 模型对比
ax1 = plt.subplot(3, 3, 1)
model_names = list(results.keys())
test_r2 = [results[name]['test_r2'] for name in model_names]
colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(model_names)))
ax1.barh(model_names, test_r2, color=colors)
ax1.set_xlabel('R² Score')
ax1.set_title('模型性能对比', fontweight='bold')
for i, v in enumerate(test_r2):
    ax1.text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=8)

# 子图2: 预测vs实际
ax2 = plt.subplot(3, 3, 2)
best_name = max(results.items(), key=lambda x: x[1]['test_r2'])[0]
y_pred_best = results[best_name]['y_pred_test']
ax2.scatter(y_test, y_pred_best, alpha=0.6, edgecolors='black', linewidth=0.5)
ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
ax2.set_xlabel('实际距离 (cm)')
ax2.set_ylabel('预测距离 (cm)')
ax2.set_title(f'{best_name} - 预测vs实际', fontweight='bold')
ax2.grid(True, alpha=0.3)

# 子图3: 残差分析
ax3 = plt.subplot(3, 3, 3)
residuals = y_test - y_pred_best
ax3.hist(residuals, bins=20, color='skyblue', edgecolor='black')
ax3.axvline(x=0, color='r', linestyle='--')
ax3.set_xlabel('残差 (cm)')
ax3.set_ylabel('频数')
ax3.set_title('残差分布', fontweight='bold')

# 子图4: 特征重要性
ax4 = plt.subplot(3, 3, 4)
if hasattr(best_rf, 'feature_importances_'):
    importances = best_rf.feature_importances_
    indices = np.argsort(importances)[-10:]
    ax4.barh(range(len(indices)), importances[indices], color='lightgreen')
    ax4.set_yticks(range(len(indices)))
    ax4.set_yticklabels([selected_features[i] for i in indices], fontsize=8)
    ax4.set_xlabel('重要性')
    ax4.set_title('特征重要性', fontweight='bold')

# 子图5: 融合对比
ax5 = plt.subplot(3, 3, 5)
ensemble_names = ['最佳单模型', 'Voting融合', '加权融合']
ensemble_scores = [results[best_name]['test_r2'], 
                   r2_score(y_test, y_pred_voting),
                   r2_score(y_test, y_pred_weighted)]
ax5.bar(ensemble_names, ensemble_scores, color=['gold', 'lightblue', 'lightgreen'])
ax5.set_ylabel('R² Score')
ax5.set_title('融合模型对比', fontweight='bold')
for i, v in enumerate(ensemble_scores):
    ax5.text(i, v + 0.01, f'{v:.4f}', ha='center', fontsize=10)

# 子图6: 时间序列
ax6 = plt.subplot(3, 3, 6)
test_idx = range(len(y_test))
ax6.plot(test_idx, y_test, 'b-', label='实际值', linewidth=2)
ax6.plot(test_idx, y_pred_best, 'r--', label='预测值', linewidth=2)
ax6.set_xlabel('样本索引')
ax6.set_ylabel('距离 (cm)')
ax6.set_title('时间序列预测', fontweight='bold')
ax6.legend()
ax6.grid(True, alpha=0.3)

# 子图7: RMSE对比
ax7 = plt.subplot(3, 3, 7)
test_rmse = [results[name]['test_rmse'] for name in model_names]
ax7.barh(model_names, test_rmse, color='lightcoral')
ax7.set_xlabel('RMSE (cm)')
ax7.set_title('RMSE对比', fontweight='bold')

# 子图8: 学习曲线
ax8 = plt.subplot(3, 3, 8)
train_sizes = np.linspace(0.1, 1.0, 10)
train_scores, test_scores = [], []
for size in train_sizes:
    n = int(len(X_train) * size)
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X_train_scaled[:n], y_train[:n])
    train_scores.append(r2_score(y_train[:n], model.predict(X_train_scaled[:n])))
    test_scores.append(r2_score(y_test, model.predict(X_test_scaled)))
ax8.plot(train_sizes, train_scores, 'o-', label='训练集')
ax8.plot(train_sizes, test_scores, 'o-', label='测试集')
ax8.set_xlabel('训练数据比例')
ax8.set_ylabel('R²')
ax8.set_title('学习曲线', fontweight='bold')
ax8.legend()
ax8.grid(True, alpha=0.3)

# 子图9: 总结
ax9 = plt.subplot(3, 3, 9)
ax9.axis('off')
summary = f"""
HC-SR04机器学习总结

特征工程:
  • 时间特征: sin/cos编码
  • 统计特征: mean/std/ema
  • 滞后特征: lag1/3/5
  • 物理特征: 声速补偿

最佳模型: {best_name}
  R²={results[best_name]['test_r2']:.4f}
  RMSE={results[best_name]['test_rmse']:.4f}cm

融合性能:
  Voting R²={r2_score(y_test, y_pred_voting):.4f}
  加权 R²={r2_score(y_test, y_pred_weighted):.4f}

超参数优化:
  最佳: {grid_search.best_params_}
"""
ax9.text(0.1, 0.95, summary, transform=ax9.transAxes, fontsize=9,
        verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.suptitle('HC-SR04超声波传感器 - 机器学习完整流程', fontsize=16, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.96])

plt.savefig(f'{output_dir}\\HC-SR04机器学习结果.png', dpi=150, bbox_inches='tight')
plt.savefig(f'{output_dir}\\HC-SR04机器学习结果_高清.png', dpi=300, bbox_inches='tight')
plt.show()

# ==================== 9. 保存结果 ====================
print("\n【步骤9】保存结果...")

results_df = pd.DataFrame({
    '模型': list(results.keys()),
    '测试R2': [results[name]['test_r2'] for name in results.keys()],
    '测试RMSE': [results[name]['test_rmse'] for name in results.keys()],
    '测试MAE': [results[name]['test_mae'] for name in results.keys()]
})
results_df.to_excel(f'{output_dir}\\模型性能对比.xlsx', index=False)

predictions_df = pd.DataFrame({
    '实际值': y_test.values if hasattr(y_test, 'values') else y_test,
    '最佳模型': y_pred_best,
    'Voting融合': y_pred_voting,
    '加权融合': y_pred_weighted
})
predictions_df.to_excel(f'{output_dir}\\预测结果.xlsx', index=False)

print(f"  结果已保存到: {output_dir}")

print("\n" + "=" * 80)
print("HC-SR04机器学习演示完成！")
print("=" * 80)
