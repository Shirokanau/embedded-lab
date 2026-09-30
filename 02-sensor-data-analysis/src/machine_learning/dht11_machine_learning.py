#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实验三 基于DHT11温湿度传感器的机器学习建模与预测
最终完整版：修复所有运行报错，Python3.12兼容，内置模拟数据
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings

# 全局配置
warnings.filterwarnings("ignore")
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False
np.random.seed(42)

# 路径与参数
OUTPUT_FOLDER = r"D:\高校专项\实验项目\实验三\experiment3_result"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
TEST_SIZE = 0.2
TARGET = "Temperature"

# ===================== 1. 内置生成DHT11时序数据 =====================
def generate_dht11_data():
    sample_num = 300
    time_series = pd.date_range(start="2026-04-22 09:00:00", periods=sample_num, freq="S")
    temp = 20 + 0.5 * np.sin(np.arange(sample_num)/50) + np.random.normal(0, 0.3, sample_num)
    humidity = 55 - 0.3 * (temp - 20) + np.random.normal(0, 1.2, sample_num)
    temp = np.clip(temp, 18, 24)
    humidity = np.clip(humidity, 35, 65)

    df = pd.DataFrame({
        "Time": time_series,
        "Temperature": np.round(temp, 2),
        "Humidity": np.round(humidity, 2)
    })
    print("【内置生成DHT11模拟数据，共300条】")
    return df

# ===================== 2. 数据预处理 =====================
def load_and_preprocess():
    df = generate_dht11_data()
    df = df.sort_values("Time").reset_index(drop=True)
    # 缺失值线性插值
    df = df.interpolate(method="linear", limit_direction="both")
    # 仅对温湿度做3-Sigma异常值过滤
    for col in ["Temperature", "Humidity"]:
        mean_val = df[col].mean()
        std_val = df[col].std()
        df = df[(df[col] >= mean_val - 3 * std_val) & (df[col] <= mean_val + 3 * std_val)]
    print(f"清洗后有效数据条数：{len(df)}")
    return df

# ===================== 3. 特征工程 =====================
def feature_construct(df):
    df_fe = df.copy()
    # 时间特征
    df_fe["Hour"] = df_fe["Time"].dt.hour
    df_fe["Minute"] = df_fe["Time"].dt.minute
    total_sec = df_fe["Hour"] * 3600 + df_fe["Minute"]
    df_fe["Time_Sin"] = np.sin(2 * np.pi * total_sec / 86400)
    df_fe["Time_Cos"] = np.cos(2 * np.pi * total_sec / 86400)
    # 滑动统计特征
    window = 10
    df_fe["Temp_Mean"] = df_fe[TARGET].rolling(window=window, min_periods=1).mean()
    df_fe["Temp_Std"] = df_fe[TARGET].rolling(window=window, min_periods=1).std()
    df_fe["Hum_Mean"] = df_fe["Humidity"].rolling(window=window, min_periods=1).mean()
    # 滞后特征
    for lag in [1, 3, 5]:
        df_fe[f"Temp_Lag{lag}"] = df_fe[TARGET].shift(lag)
        df_fe[f"Hum_Lag{lag}"] = df_fe["Humidity"].shift(lag)
    df_fe.fillna(method="bfill", inplace=True)
    # 交互特征
    df_fe["Temp_Hum"] = df_fe[TARGET] * df_fe["Humidity"]
    df_fe["Temp_Square"] = df_fe[TARGET] ** 2
    df_fe["Hum_Square"] = df_fe["Humidity"] ** 2
    df_fe["Temp_Hum_Ratio"] = df_fe[TARGET] / (df_fe["Humidity"] + 1)
    print(f"特征工程完成，总特征维度：{df_fe.shape[1]}")
    return df_fe

# ===================== 4. 特征选择 + 数据集划分 =====================
def feature_select_and_split(df):
    feature_cols = [col for col in df.columns if col not in ["Time", TARGET]]
    X = df[feature_cols]
    y = df[TARGET]
    print(f"原始特征数：{len(feature_cols)}")
    # 特征筛选
    selector = SelectKBest(score_func=f_regression, k=15)
    X_selected = selector.fit_transform(X, y)
    select_features = X.columns[selector.get_support()].tolist()
    print(f"筛选后特征数：{len(select_features)}")
    # 数据集划分
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=TEST_SIZE, random_state=42
    )
    print(f"训练集：{len(X_train)}，测试集：{len(X_test)}")
    # 标准化
    scaler = StandardScaler()
    X_train_sca = scaler.fit_transform(X_train)
    X_test_sca = scaler.transform(X_test)
    return X_train_sca, X_test_sca, y_train, y_test, select_features

# ===================== 5. 多模型训练与评估 =====================
def train_models(X_train, X_test, y_train, y_test):
    model_dict = {
        "线性回归": LinearRegression(),
        "岭回归": Ridge(alpha=1.0),
        "Lasso回归": Lasso(alpha=0.1),
        "随机森林": RandomForestRegressor(n_estimators=100, random_state=42),
        "梯度提升": GradientBoostingRegressor(n_estimators=100, random_state=42),
        "支持向量机(SVR)": SVR(kernel="rbf"),
        "多层感知机(MLP)": MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
    }
    model_result = {}

    for name, model in model_dict.items():
        print(f"\n------ 训练 {name} ------")
        model.fit(X_train, y_train)
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        cv_r2 = cross_val_score(model, X_train, y_train, cv=5, scoring="r2").mean()

        model_result[name] = {
            "model": model,
            "train_rmse": train_rmse,
            "test_rmse": test_rmse,
            "train_mae": train_mae,
            "test_mae": test_mae,
            "train_r2": train_r2,
            "test_r2": test_r2,
            "cv_r2": cv_r2,
            "y_pred": y_test_pred
        }
        print(f"训练集 R2:{train_r2:.4f} RMSE:{train_rmse:.4f}")
        print(f"测试集 R2:{test_r2:.4f} RMSE:{test_rmse:.4f}")
        print(f"5折交叉验证 R2:{cv_r2:.4f}")
    return model_result

# ===================== 6. 模型融合（修复变量作用域问题） =====================
def model_ensemble(results, y_test):
    # 按测试集R2降序排序
    sorted_items = sorted(results.items(), key=lambda x: x[1]["test_r2"], reverse=True)
    top3_items = sorted_items[:3]

    top3_names = []
    top3_preds = []
    top3_scores = []
    for name, res_dict in top3_items:
        top3_names.append(name)
        top3_preds.append(res_dict["y_pred"])
        top3_scores.append(res_dict["test_r2"])

    print(f"\n【模型融合】选取Top3模型：{top3_names}")

    # 简单平均融合
    avg_pred = np.mean(top3_preds, axis=0)
    avg_r2 = r2_score(y_test, avg_pred)
    avg_rmse = np.sqrt(mean_squared_error(y_test, avg_pred))

    # 加权融合（提前定义变量，杜绝UnboundLocalError）
    weights = np.array(top3_scores)
    weights = weights / np.sum(weights)
    weight_pred = np.average(top3_preds, axis=0, weights=weights)
    weight_r2 = r2_score(y_test, weight_pred)
    weight_rmse = np.sqrt(mean_squared_error(y_test, weight_pred))

    print(f"简单平均融合：R2={avg_r2:.4f} RMSE={avg_rmse:.4f}")
    print(f"加权融合(权重:{np.round(weights,4)})：R2={weight_r2:.4f} RMSE={weight_rmse:.4f}")

    return {
        "avg_pred": avg_pred,
        "weight_pred": weight_pred,
        "avg_r2": avg_r2,
        "weight_r2": weight_r2
    }

# ===================== 7. 可视化 & 结果导出 =====================
def visualize_and_save(results, ensemble, y_test):
    # 导出模型指标表
    res_list = []
    for name, val in results.items():
        res_list.append({
            "模型名称": name,
            "训练集R2": val["train_r2"],
            "测试集R2": val["test_r2"],
            "测试集RMSE": val["test_rmse"],
            "测试集MAE": val["test_mae"],
            "交叉验证R2": val["cv_r2"]
        })
    res_df = pd.DataFrame(res_list)
    res_df.to_excel(os.path.join(OUTPUT_FOLDER, "模型性能对比表.xlsx"), index=False)

    # 导出预测结果
    best_name = max(results, key=lambda k: results[k]["test_r2"])
    pred_df = pd.DataFrame({
        "真实温度": y_test.values,
        "最优单模型预测": results[best_name]["y_pred"],
        "简单融合预测": ensemble["avg_pred"],
        "加权融合预测": ensemble["weight_pred"]
    })
    pred_df.to_excel(os.path.join(OUTPUT_FOLDER, "预测结果表.xlsx"))

    # 绘图（标准subplot三参数格式）
    fig = plt.figure(figsize=(16, 12))
    # 子图1：各模型测试集R²对比
    ax1 = plt.subplot(2, 2, 1)
    names = list(results.keys())
    r2_list = [results[n]["test_r2"] for n in names]
    ax1.barh(names, r2_list, color="steelblue")
    ax1.set_xlabel("决定系数 R²")
    ax1.set_title("各模型测试集 R² 对比")

    # 子图2：各模型测试集RMSE对比
    ax2 = plt.subplot(2, 2, 2)
    rmse_list = [results[n]["test_rmse"] for n in names]
    ax2.barh(names, rmse_list, color="lightcoral")
    ax2.set_xlabel("均方根 RMSE(℃)")
    ax2.set_title("各模型测试集 RMSE 对比")

    # 子图3：最优模型 真实值 vs 预测值
    ax3 = plt.subplot(2, 2, 3)
    best_pred = results[best_name]["y_pred"]
    ax3.scatter(y_test, best_pred, alpha=0.7)
    ax3.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
    ax3.set_xlabel("真实温度(℃)")
    ax3.set_ylabel("预测温度(℃)")
    ax3.set_title(f"{best_name} 真实值与预测值对比")

    # 子图4：融合模型对比
    ax4 = plt.subplot(2, 2, 4)
    fuse_name = ["最优单模型", "简单融合", "加权融合"]
    fuse_r2 = [results[best_name]["test_r2"], ensemble["avg_r2"], ensemble["weight_r2"]]
    ax4.bar(fuse_name, fuse_r2, color=["gold", "lightblue", "lightgreen"])
    ax4.set_ylabel("决定系数 R²")
    ax4.set_title("融合模型性能对比")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FOLDER, "机器学习结果图.png"), dpi=300)
    plt.show()
    print(f"\n结果图表、表格已保存至 {OUTPUT_FOLDER} 文件夹")

# ===================== 主程序入口 =====================
if __name__ == "__main__":
    print("="*70)
    print("实验三 基于DHT11温湿度传感器的机器学习建模与预测")
    print("="*70)

    df_raw = load_and_preprocess()
    df_fe = feature_construct(df_raw)
    X_train, X_test, y_train, y_test, _ = feature_select_and_split(df_fe)
    model_res = train_models(X_train, X_test, y_train, y_test)
    ensemble_res = model_ensemble(model_res, y_test)
    visualize_and_save(model_res, ensemble_res, y_test)

    print("\n" + "="*70)
    print("实验全部执行完毕！")
    print("="*70)