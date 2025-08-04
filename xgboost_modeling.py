import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("=== 二手车价格预测 - XGBoost建模 ===")

# 1. 数据加载
print("\n1. 加载数据...")
train_df = pd.read_csv('used_car_train_20200313.csv', sep=' ')
test_df = pd.read_csv('used_car_testB_20200421.csv', sep=' ')

print(f"训练集形状: {train_df.shape}")
print(f"测试集形状: {test_df.shape}")

# 2. 数据预处理
print("\n2. 数据预处理...")

def preprocess_data(df, is_train=True):
    """数据预处理函数"""
    df_processed = df.copy()
    
    # 处理缺失值
    # 数值型变量用中位数填充
    numeric_cols = ['model', 'bodyType', 'fuelType', 'gearbox']
    for col in numeric_cols:
        if col in df_processed.columns:
            df_processed[col] = df_processed[col].fillna(df_processed[col].median())
    
    # 处理notRepairedDamage列
    if 'notRepairedDamage' in df_processed.columns:
        # 将'-'替换为NaN，然后填充为0
        df_processed['notRepairedDamage'] = df_processed['notRepairedDamage'].replace('-', np.nan)
        df_processed['notRepairedDamage'] = df_processed['notRepairedDamage'].fillna(0)
        df_processed['notRepairedDamage'] = df_processed['notRepairedDamage'].astype(float)
    
    # 特征工程
    # 1. 注册年份特征
    df_processed['regYear'] = df_processed['regDate'].astype(str).str[:4].astype(int)
    df_processed['regMonth'] = df_processed['regDate'].astype(str).str[4:6].astype(int)
    
    # 2. 创建日期特征
    df_processed['creatYear'] = df_processed['creatDate'].astype(str).str[:4].astype(int)
    df_processed['creatMonth'] = df_processed['creatDate'].astype(str).str[4:6].astype(int)
    
    # 3. 车龄特征
    df_processed['carAge'] = df_processed['creatYear'] - df_processed['regYear']
    
    # 4. 功率分组
    df_processed['powerGroup'] = pd.cut(df_processed['power'], 
                                       bins=[0, 100, 150, 200, 300, 1000, 20000], 
                                       labels=[0, 1, 2, 3, 4, 5], 
                                       include_lowest=True).astype(int)
    
    # 5. 公里数分组
    df_processed['kilometerGroup'] = pd.cut(df_processed['kilometer'], 
                                           bins=[0, 5, 10, 15, 20, 25, 30], 
                                           labels=[0, 1, 2, 3, 4, 5], 
                                           include_lowest=True).astype(int)
    
    # 6. 价格对数变换（仅对训练集）
    if is_train and 'price' in df_processed.columns:
        df_processed['price_log'] = np.log1p(df_processed['price'])
    
    return df_processed

# 预处理训练集和测试集
train_processed = preprocess_data(train_df, is_train=True)
test_processed = preprocess_data(test_df, is_train=False)

print("数据预处理完成")

# 3. 特征选择
print("\n3. 特征选择...")

# 根据EDA分析，选择重要特征
feature_cols = [
    'name', 'regDate', 'model', 'brand', 'bodyType', 'fuelType', 'gearbox',
    'power', 'kilometer', 'notRepairedDamage', 'regionCode', 'seller', 'offerType',
    'creatDate', 'regYear', 'regMonth', 'creatYear', 'creatMonth', 'carAge',
    'powerGroup', 'kilometerGroup'
] + [f'v_{i}' for i in range(15)]  # 匿名特征变量

# 确保所有特征都存在
available_features = [col for col in feature_cols if col in train_processed.columns]
print(f"使用特征数量: {len(available_features)}")
print(f"特征列表: {available_features}")

# 4. 数据分割
print("\n4. 数据分割...")
X = train_processed[available_features]
y = train_processed['price']

# 80%训练，20%验证
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=None
)

print(f"训练集大小: {X_train.shape}")
print(f"验证集大小: {X_val.shape}")

# 5. 模型训练
print("\n5. XGBoost模型训练...")

# 设置XGBoost参数
xgb_params = {
    'objective': 'reg:squarederror',
    'n_estimators': 1000,
    'max_depth': 6,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'n_jobs': -1,
    'early_stopping_rounds': 50
}

# 训练模型
model = xgb.XGBRegressor(**xgb_params)
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=100
)

# 6. 模型评估
print("\n6. 模型评估...")

# 训练集预测
y_train_pred = model.predict(X_train)
y_val_pred = model.predict(X_val)

# 计算评估指标
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
train_mae = mean_absolute_error(y_train, y_train_pred)
val_mae = mean_absolute_error(y_val, y_val_pred)
train_r2 = r2_score(y_train, y_train_pred)
val_r2 = r2_score(y_val, y_val_pred)

print("=== 模型性能评估 ===")
print(f"训练集 RMSE: {train_rmse:.2f}")
print(f"验证集 RMSE: {val_rmse:.2f}")
print(f"训练集 MAE: {train_mae:.2f}")
print(f"验证集 MAE: {val_mae:.2f}")
print(f"训练集 R²: {train_r2:.4f}")
print(f"验证集 R²: {val_r2:.4f}")

# 7. 特征重要性分析
print("\n7. 特征重要性分析...")
feature_importance = pd.DataFrame({
    'feature': available_features,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("前10个重要特征:")
print(feature_importance.head(10))

# 8. 预测测试集
print("\n8. 预测测试集...")
X_test = test_processed[available_features]
test_predictions = model.predict(X_test)

# 创建预测结果DataFrame，确保格式正确
test_results = pd.DataFrame({
    'SaleID': test_processed['SaleID'],
    'price': test_predictions
})

# 确保price列为整数类型
test_results['price'] = test_results['price'].round().astype(int)

# 保存预测结果
test_results.to_csv('test_predictions.csv', index=False)
print(f"预测结果已保存到 test_predictions.csv")
print(f"预测样本数量: {len(test_results)}")
print(f"预测价格范围: {test_results['price'].min()} - {test_results['price'].max()}")
print(f"预测价格均值: {test_results['price'].mean():.2f}")

# 显示前10个预测结果
print("\n前10个预测结果:")
print(test_results.head(10))

# 打印验证集MAE
print(f"\n=== 验证集性能评估 ===")
print(f"验证集 MAE: {val_mae:.2f}")
print(f"验证集 RMSE: {val_rmse:.2f}")
print(f"验证集 R²: {val_r2:.4f}")

# 9. 可视化分析
print("\n9. 生成可视化图表...")

# 创建图表
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('XGBoost模型性能分析', fontsize=16)

# 1. 预测vs实际值散点图
axes[0, 0].scatter(y_val, y_val_pred, alpha=0.5, s=1)
axes[0, 0].plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--', lw=2)
axes[0, 0].set_xlabel('实际价格')
axes[0, 0].set_ylabel('预测价格')
axes[0, 0].set_title('验证集: 预测vs实际')

# 2. 残差图
residuals = y_val - y_val_pred
axes[0, 1].scatter(y_val_pred, residuals, alpha=0.5, s=1)
axes[0, 1].axhline(y=0, color='r', linestyle='--')
axes[0, 1].set_xlabel('预测价格')
axes[0, 1].set_ylabel('残差')
axes[0, 1].set_title('残差分布')

# 3. 特征重要性
top_features = feature_importance.head(10)
axes[1, 0].barh(range(len(top_features)), top_features['importance'])
axes[1, 0].set_yticks(range(len(top_features)))
axes[1, 0].set_yticklabels(top_features['feature'])
axes[1, 0].set_xlabel('重要性')
axes[1, 0].set_title('特征重要性 (前10)')

# 4. 预测分布
axes[1, 1].hist(test_predictions, bins=50, alpha=0.7, color='green', edgecolor='black')
axes[1, 1].set_xlabel('预测价格')
axes[1, 1].set_ylabel('频数')
axes[1, 1].set_title('测试集预测价格分布')

plt.tight_layout()
plt.savefig('xgboost_model_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

# 10. 保存模型和结果
print("\n10. 保存模型和结果...")

# 保存模型
import joblib
joblib.dump(model, 'xgboost_model.pkl')
print("模型已保存为 xgboost_model.pkl")

# 保存评估结果
results_summary = {
    'train_rmse': train_rmse,
    'val_rmse': val_rmse,
    'train_mae': train_mae,
    'val_mae': val_mae,
    'train_r2': train_r2,
    'val_r2': val_r2,
    'feature_importance': feature_importance.to_dict('records')
}

import json
with open('model_results.json', 'w', encoding='utf-8') as f:
    json.dump(results_summary, f, ensure_ascii=False, indent=2)

print("模型结果已保存为 model_results.json")

# 11. 生成Markdown报告
print("\n11. 生成Markdown报告...")

md_content = []
md_content.append("# 二手车价格预测 - XGBoost建模报告\n")
md_content.append(f"**建模时间**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

md_content.append("## 1. 数据概况\n")
md_content.append(f"- **训练集大小**: {train_df.shape[0]} 行, {train_df.shape[1]} 列")
md_content.append(f"- **测试集大小**: {test_df.shape[0]} 行, {test_df.shape[1]} 列")
md_content.append(f"- **使用特征数量**: {len(available_features)}")
md_content.append(f"- **训练集样本**: {X_train.shape[0]}")
md_content.append(f"- **验证集样本**: {X_val.shape[0]}\n")

md_content.append("## 2. 模型性能\n")
md_content.append("| 指标 | 训练集 | 验证集 |")
md_content.append("|------|--------|--------|")
md_content.append(f"| RMSE | {train_rmse:.2f} | {val_rmse:.2f} |")
md_content.append(f"| MAE | {train_mae:.2f} | {val_mae:.2f} |")
md_content.append(f"| R² | {train_r2:.4f} | {val_r2:.4f} |\n")

md_content.append("## 3. 特征重要性 (前10)\n")
md_content.append("| 排名 | 特征名 | 重要性 |")
md_content.append("|------|--------|--------|")
for i, (_, row) in enumerate(feature_importance.head(10).iterrows(), 1):
    md_content.append(f"| {i} | {row['feature']} | {row['importance']:.4f} |")

md_content.append("\n## 4. 模型配置\n")
md_content.append("```")
md_content.append(str(xgb_params))
md_content.append("```\n")

md_content.append("## 5. 预测结果\n")
md_content.append(f"- **测试集预测样本数**: {len(test_results)}")
md_content.append(f"- **预测价格范围**: {test_predictions.min():.0f} - {test_predictions.max():.0f}")
md_content.append(f"- **预测价格均值**: {test_predictions.mean():.2f}")
md_content.append(f"- **预测价格中位数**: {np.median(test_predictions):.2f}")

# 保存Markdown报告
with open('xgboost_modeling_report.md', 'w', encoding='utf-8') as f:
    f.write('\n'.join(md_content))

print("Markdown报告已保存为 xgboost_modeling_report.md")
print("\n=== 建模完成 ===") 