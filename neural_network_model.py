import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("二手车价格预测 - 神经网络模型")
print("=" * 60)

# 1. 数据加载
print("\n1. 数据加载...")
train_df = pd.read_csv('used_car_train_20200313.csv', sep=' ')
test_df = pd.read_csv('used_car_testB_20200421.csv', sep=' ')

print(f"训练集: {train_df.shape}")
print(f"测试集: {test_df.shape}")

# 2. 数据预处理
print("\n2. 数据预处理...")

def preprocess_data(df, is_train=True):
    df_processed = df.copy()
    
    # 处理缺失值
    numeric_features = ['model', 'bodyType', 'fuelType', 'gearbox']
    for feature in numeric_features:
        if feature in df_processed.columns:
            df_processed[feature] = df_processed[feature].fillna(df_processed[feature].median())
    
    # 处理notRepairedDamage特殊缺失值
    df_processed['notRepairedDamage'] = df_processed['notRepairedDamage'].replace('-', '0.0')
    df_processed['notRepairedDamage'] = df_processed['notRepairedDamage'].astype(float)
    
    # 创建车龄特征
    df_processed['regDate_parsed'] = pd.to_datetime(df_processed['regDate'], format='%Y%m%d', errors='coerce')
    df_processed['creatDate_parsed'] = pd.to_datetime(df_processed['creatDate'], format='%Y%m%d', errors='coerce')
    df_processed['car_age'] = (df_processed['creatDate_parsed'] - df_processed['regDate_parsed']).dt.days / 365.25
    df_processed['car_age'] = df_processed['car_age'].fillna(df_processed['car_age'].median())
    
    # 创建功率密度特征
    df_processed['power_density'] = df_processed['power'] / (df_processed['car_age'] + 1)
    
    # 创建新车判断特征（一年内购入的车为新车）
    df_processed['is_newage'] = (df_processed['car_age'] <= 1).astype(int)
    
    # 创建每岁车龄的平均行驶里程特征
    df_processed['km_per_age'] = df_processed['kilometer'] / (df_processed['car_age'] + 1)
    
    # 选择特征
    feature_columns = [
        'power', 'kilometer', 'model', 'brand', 'bodyType', 'fuelType', 'gearbox',
        'notRepairedDamage', 'regionCode', 'seller', 'offerType', 'car_age', 'power_density', 'is_newage', 'km_per_age'
    ] + [f'v_{i}' for i in range(15)]
    
    available_features = [col for col in feature_columns if col in df_processed.columns]
    X = df_processed[available_features]
    
    if is_train:
        y = df_processed['price']
        return X, y
    else:
        return X

# 预处理训练数据
X_train_full, y_train_full = preprocess_data(train_df, is_train=True)
print(f"特征数量: {X_train_full.shape[1]}")

# 3. 数据分割
print("\n3. 数据分割...")
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.2, random_state=42
)

print(f"训练集大小: {X_train.shape}")
print(f"验证集大小: {X_val.shape}")

# 4. 特征标准化
print("\n4. 特征标准化...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# 转换为numpy数组
X_train_array = np.array(X_train_scaled, dtype=np.float32)
X_val_array = np.array(X_val_scaled, dtype=np.float32)
y_train_array = np.array(y_train, dtype=np.float32)
y_val_array = np.array(y_val, dtype=np.float32)

# 5. 构建2层神经网络
print("\n5. 构建神经网络模型...")
input_dim = X_train_array.shape[1]

model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(input_dim,)),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(1, activation='linear')
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='mae',
    metrics=['mae']
)

print("神经网络结构:")
model.summary()

# 6. 模型训练
print("\n6. 神经网络模型训练...")
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=30,
    restore_best_weights=True
)

history = model.fit(
    X_train_array, y_train_array,
    validation_data=(X_val_array, y_val_array),
    epochs=200,
    batch_size=32,
    callbacks=[early_stopping],
    verbose=1
)

# 7. 模型评估
print("\n7. 模型评估...")
y_train_pred = model.predict(X_train_array).flatten()
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
train_r2 = r2_score(y_train, y_train_pred)
train_mae = mean_absolute_error(y_train, y_train_pred)

y_val_pred = model.predict(X_val_array).flatten()
val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
val_r2 = r2_score(y_val, y_val_pred)
val_mae = mean_absolute_error(y_val, y_val_pred)

print("训练集性能:")
print(f"RMSE: {train_rmse:.2f}")
print(f"R²: {train_r2:.4f}")
print(f"MAE: {train_mae:.2f}")

print("\n验证集性能:")
print(f"RMSE: {val_rmse:.2f}")
print(f"R²: {val_r2:.4f}")
print(f"MAE: {val_mae:.2f}")

# 8. 特征重要性分析
print("\n8. 特征重要性分析...")
first_layer_weights = model.layers[0].get_weights()[0]
feature_importance = np.mean(np.abs(first_layer_weights), axis=1)

feature_importance_df = pd.DataFrame({
    'feature': X_train_full.columns,
    'importance': feature_importance
})
feature_importance_df = feature_importance_df.sort_values('importance', ascending=False)

print("特征重要性 (前10):")
print(feature_importance_df.head(10))

# 9. 测试集预测
print("\n9. 测试集预测...")
X_test = preprocess_data(test_df, is_train=False)
X_test_scaled = scaler.transform(X_test)
X_test_array = np.array(X_test_scaled, dtype=np.float32)

test_predictions = model.predict(X_test_array).flatten()
test_predictions = np.maximum(test_predictions, 0)

# 10. 保存预测结果
print("\n10. 保存预测结果...")
results_df = pd.DataFrame({
    'SaleID': test_df['SaleID'],
    'price': test_predictions
})

results_df.to_csv('neural_network_predictions.csv', index=False)
print(f"预测结果已保存到 neural_network_predictions.csv")
print(f"预测样本数: {len(results_df)}")

# 11. 预测结果统计
print("\n11. 预测结果统计...")
print("预测价格统计:")
print(results_df['price'].describe())

print(f"\n预测价格范围: {results_df['price'].min():.2f} - {results_df['price'].max():.2f}")
print(f"预测价格均值: {results_df['price'].mean():.2f}")
print(f"预测价格中位数: {results_df['price'].median():.2f}")

# 12. 模型总结
print("\n12. 模型总结...")
print("=" * 60)
print("神经网络模型训练完成!")
print(f"使用特征数量: {X_train_full.shape[1]}")
print(f"训练样本数: {X_train.shape[0]}")
print(f"验证样本数: {X_val.shape[0]}")
print(f"测试样本数: {len(results_df)}")
print(f"验证集R²: {val_r2:.4f}")
print(f"验证集RMSE: {val_rmse:.2f}")

print("\n神经网络结构:")
print("- 输入层: {} 个特征".format(input_dim))
print("- 第一层: 128 个神经元 (ReLU激活)")
print("- Dropout: 0.3")
print("- 第二层: 64 个神经元 (ReLU激活)")
print("- Dropout: 0.2")
print("- 输出层: 1 个神经元 (线性激活)")

print("\n主要特征 (按重要性排序):")
for i, row in feature_importance_df.head(5).iterrows():
    print(f"{row['feature']}: {row['importance']:.4f}")

print("\n预测结果已保存到 neural_network_predictions.csv")
print("文件格式: SaleID, price") 