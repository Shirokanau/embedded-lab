#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HC-SR04 超声波传感器 机器学习（简易版）
纯读取Excel、修复语法错误 + 全中文列名适配
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
import warnings

# 全局配置
warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
np.random.seed(42)

# 路径配置
DATA_PATH = r'D:\高校专项\实验项目\实验三\05_HCSR04机器学习\HC-SR04预处理结果数据.xlsx'
OUTPUT_DIR = r"D:\高校专项\实验项目\实验三\05_HCSR04机器学习"
os.makedirs(OUTPUT_DIR, exist_ok=True)
TEST_SIZE = 0.2

# ===================== 特征工程 =====================
def build_features(df):
    # 全使用中文列名，匹配Excel
    df["time_sin"] = np.sin(2 * np.pi * df["时间"] / 300)
    df["time_cos"] = np.cos(2 * np.pi * df["时间"] / 300)
    df["dist_mean_5"] = df["距离"].rolling(5, min_periods=1).mean()
    df["dist_std_5"] = df["距离"].rolling(5, min_periods=1).std()
    df["dist_lag1"] = df["距离"].shift(1)
    df["dist_lag3"] = df["距离"].shift(3)
    df["sound_speed"] = 331.4 + 12
    df["dist_correct"] = df["距离"] * (331.4 / df["sound_speed"])
    df["motion"] = np.where(np.abs(df["速度"]) > 0.1, 1, 0)
    return df

# ===================== 主流程 =====================
if __name__ == "__main__":
    print("=" * 70)
    print("HC-SR04 超声波传感器 机器学习（简易版）")
    print("=" * 70)

    # 读取Excel数据
    print("\n【读取预处理数据】")
    df = pd.read_excel(DATA_PATH, engine="openpyxl")
    print(f"成功加载数据，总条数：{len(df)}")
    print(f"表格列名：{list(df.columns)}")

    # 构造特征
    df = build_features(df)

    # 特征列（程序内部衍生列，英文无影响）
    feature_cols = ["velocity","acceleration","time_sin","time_cos",
                    "dist_mean_5","dist_std_5","dist_lag1","dist_lag3",
                    "dist_correct","motion"]
    # 对应Excel原始中文列
    df.rename(columns={"速度":"velocity", "加速度":"acceleration"}, inplace=True)

    X = df[feature_cols]
    y = df["真实距离"]

    # 填充缺失值 NaN
    imputer = SimpleImputer(strategy="mean")
    X = imputer.fit_transform(X)

    # 数据集划分
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=42)
    # 标准化
    scaler = StandardScaler()
    X_train_sca = scaler.fit_transform(X_train)
    X_test_sca = scaler.transform(X_test)

    # 模型定义
    models = {
        "线性回归": LinearRegression(),
        "岭回归": Ridge(alpha=1.0),
        "随机森林": RandomForestRegressor(n_estimators=100, random_state=42),
        "梯度提升": GradientBoostingRegressor(n_estimators=100, random_state=42),
        "SVR": SVR(kernel="rbf"),
        "多层感知机": MLPRegressor(hidden_layer_sizes=(100,50), max_iter=2000, random_state=42)
    }
    results = {}

    # 模型训练
    print("\n【模型训练】")
    for name, model in models.items():
        model.fit(X_train_sca, y_train)
        y_pred = model.predict(X_test_sca)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        results[name] = {"r2": r2, "rmse": rmse, "pred": y_pred}
        print(f"{name}  R²:{r2:.4f}  RMSE:{rmse:.4f} cm")

    # 模型融合
    print("\n【模型融合】")
    top3 = sorted(results.items(), key=lambda x: x[1]["r2"], reverse=True)[:3]
    estimators = [(n, models[n]) for n, _ in top3]
    vote_model = VotingRegressor(estimators)
    vote_model.fit(X_train_sca, y_train)
    vote_pred = vote_model.predict(X_test_sca)
    vote_r2 = r2_score(y_test, vote_pred)
    vote_rmse = np.sqrt(mean_squared_error(y_test, vote_pred))

    # 加权融合
    weights = np.array([r['r2'] for _, r in top3])
    weights = weights / weights.sum()
    pred_arr = np.array([r['pred'] for _, r in top3])
    weight_pred = np.average(pred_arr, axis=0, weights=weights)
    weight_r2 = r2_score(y_test, weight_pred)
    weight_rmse = np.sqrt(mean_squared_error(y_test, weight_pred))

    print(f"投票融合  R²:{vote_r2:.4f}  RMSE:{vote_rmse:.4f}")
    print(f"加权融合  R²:{weight_r2:.4f}  RMSE:{weight_rmse:.4f}")

    # 3×3 共9张子图可视化
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    plt.suptitle('HC-SR04超声波传感器 - 机器学习完整流程', fontsize=14, fontweight='bold')
    names = list(results.keys())
    r2_list = [results[n]["r2"] for n in names]
    rmse_list = [results[n]["rmse"] for n in names]
    best_name = max(results, key=lambda x: results[x]["r2"])
    best_pred = results[best_name]["pred"]
    lr_pred = results["线性回归"]["pred"]

    # 上排左：模型 R² 对比
    axes[0,0].barh(names, r2_list, color="steelblue")
    axes[0,0].set_xlabel("R² Score")
    axes[0,0].set_title("模型性能对比（R²）")

    # 上排中：线性回归 真实值 vs 预测值
    axes[0,1].scatter(y_test, lr_pred, alpha=0.6, color='#1f77b4')
    axes[0,1].plot([y_test.min(), y_test.max()],[y_test.min(), y_test.max()],"r--", label="理想线")
    axes[0,1].set_xlabel("真实距离 (cm)")
    axes[0,1].set_ylabel("预测距离 (cm)")
    axes[0,1].set_title("线性回归 - 预测vs实际")
    axes[0,1].legend()

    # 上排右：残差分布
    residuals = y_test - best_pred
    axes[0,2].hist(residuals, bins=20, color="skyblue", edgecolor="black")
    axes[0,2].axvline(0, color="r", linestyle="--")
    axes[0,2].set_xlabel("残差 (cm)")
    axes[0,2].set_ylabel("频数")
    axes[0,2].set_title("残差分布")

    # 中排左：特征重要性（随机森林）
    rf_model = models["随机森林"]
    imp = rf_model.feature_importances_
    idx = np.argsort(imp)
    axes[1,0].barh(range(len(idx)), imp[idx], color="#90EE90")
    axes[1,0].set_yticks(range(len(idx)))
    axes[1,0].set_yticklabels(feature_cols, fontsize=8)
    axes[1,0].set_xlabel("重要性")
    axes[1,0].set_title("特征重要性")

    # 中排中：融合模型对比
    fuse_names = ["最佳单模型", "投票融合", "加权融合"]
    fuse_r2 = [results[best_name]["r2"], vote_r2, weight_r2]
    axes[1,1].bar(fuse_names, fuse_r2, color=["gold", "lightblue", "lightgreen"])
    axes[1,1].set_ylabel("R² Score")
    axes[1,1].set_title("融合模型对比")

    # 中排右：时序预测曲线
    axes[1,2].plot(y_test.values, label="真实值", linewidth=2, color="blue")
    axes[1,2].plot(best_pred, label="预测值", linestyle="--", linewidth=2, color="red")
    axes[1,2].set_xlabel("样本索引")
    axes[1,2].set_ylabel("距离 (cm)")
    axes[1,2].set_title("时间序列预测")
    axes[1,2].legend()

    # 下排左：模型 RMSE 对比
    axes[2,0].barh(names, rmse_list, color="#ee9999")
    axes[2,0].set_xlabel("RMSE (cm)")
    axes[2,0].set_title("模型RMSE对比")

    # 下排中：学习曲线（线性回归）
    train_sizes = np.linspace(0.2, 1.0, 8)
    train_scores = []
    test_scores = []
    for size in train_sizes:
        split_idx = int(len(X_train_sca) * size)
        X_p = X_train_sca[:split_idx]
        y_p = y_train[:split_idx]
        lr = LinearRegression()
        lr.fit(X_p, y_p)
        train_scores.append(r2_score(y_p, lr.predict(X_p)))
        test_scores.append(r2_score(y_test, lr.predict(X_test_sca)))
    axes[2,1].plot(train_sizes, train_scores, marker="o", color="blue", label="训练集")
    axes[2,1].plot(train_sizes, test_scores, marker="o", color="orange", label="测试集")
    axes[2,1].set_xlabel("训练数据比例")
    axes[2,1].set_ylabel("R² Score")
    axes[2,1].set_title("学习曲线")
    axes[2,1].legend()

    # 下排右：结果文本汇总
    axes[2,2].axis("off")
    text = f"""HC-SR04 实验结果汇总
最优模型：{best_name}
R²: {results[best_name]['r2']:.6f}
RMSE: {results[best_name]['rmse']:.6f} cm

投票融合
R²: {vote_r2:.6f}
RMSE: {vote_rmse:.6f} cm

加权融合
R²: {weight_r2:.6f}
RMSE: {weight_rmse:.6f} cm
"""
    axes[2,2].text(0.05, 0.95, text, va="top", fontsize=10, linespacing=1.6,
                   bbox=dict(facecolor="#fff3e0", alpha=0.7))

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "HC_SR04_9图结果.png"), dpi=300)
    plt.show()

    # 保存表格
    res_df = pd.DataFrame({
        "模型": names,
        "R²": r2_list,
        "RMSE(cm)": rmse_list
    })
    res_df.to_excel(os.path.join(OUTPUT_DIR, "模型结果.xlsx"), index=False, engine="openpyxl")

    print(f"\n结果已保存至：{OUTPUT_DIR}")
    print("=" * 70)
    print("实验运行完毕！")