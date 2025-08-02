import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# 设置随机种子确保结果可重现
tf.random.set_seed(42)
np.random.seed(42)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_and_preprocess_data():
    """加载和预处理训练数据"""
    print("加载训练数据...")
    train_data = pd.read_csv('used_car_train_20200313.csv', sep=' ')
    
    print(f"原始训练数据形状: {train_data.shape}")
    print(f"原始数据列名: {train_data.columns.tolist()}")
    
    # 数据预处理
    print("\n开始数据预处理...")
    
    # 1. 处理缺失值
    print("处理缺失值...")
    # 用众数填充分类特征的缺失值
    categorical_features = ['model', 'brand', 'bodyType', 'fuelType', 'gearbox', 'notRepairedDamage']
    for feature in categorical_features:
        if feature in train_data.columns:
            mode_value = train_data[feature].mode()[0] if not train_data[feature].mode().empty else 0
            train_data[feature].fillna(mode_value, inplace=True)
    
    # 用中位数填充数值特征的缺失值
    numeric_features = ['power', 'kilometer'] + [f'v_{i}' for i in range(15)]
    for feature in numeric_features:
        if feature in train_data.columns:
            median_value = train_data[feature].median()
            train_data[feature].fillna(median_value, inplace=True)
    
    # 2. 处理异常值
    print("处理异常值...")
    
    # 处理负价格
    if 'price' in train_data.columns:
        print(f"发现 {(train_data['price'] < 0).sum()} 个负价格值")
        train_data = train_data[train_data['price'] >= 0]
    
    # 处理异常功率值
    if 'power' in train_data.columns:
        # 转换为数值类型
        train_data['power'] = pd.to_numeric(train_data['power'], errors='coerce')
        train_data['power'].fillna(train_data['power'].median(), inplace=True)
        
        # 处理异常功率值 (通常在0-1000之间)
        train_data.loc[train_data['power'] > 1000, 'power'] = train_data['power'].median()
        train_data.loc[train_data['power'] < 0, 'power'] = train_data['power'].median()
    
    # 处理异常里程值
    if 'kilometer' in train_data.columns:
        train_data['kilometer'] = pd.to_numeric(train_data['kilometer'], errors='coerce')
        train_data['kilometer'].fillna(train_data['kilometer'].median(), inplace=True)
        
        # 处理异常里程值
        train_data.loc[train_data['kilometer'] > 1000000, 'kilometer'] = train_data['kilometer'].median()
        train_data.loc[train_data['kilometer'] < 0, 'kilometer'] = train_data['kilometer'].median()
    
    # 3. 处理分类特征
    print("编码分类特征...")
    label_encoders = {}
    
    # 对brand进行One-Hot编码
    if 'brand' in train_data.columns:
        print("对brand进行One-Hot编码...")
        brand_dummies = pd.get_dummies(train_data['brand'], prefix='brand')
        print(f"brand特征展开为 {brand_dummies.shape[1]} 个特征")
        # 将One-Hot编码的列添加到数据中
        train_data = pd.concat([train_data, brand_dummies], axis=1)
        # 保存brand的唯一值供测试时使用
        label_encoders['brand_columns'] = brand_dummies.columns.tolist()
    
    # 对其他分类特征使用LabelEncoder
    other_categorical_features = [f for f in categorical_features if f != 'brand']
    for feature in other_categorical_features:
        if feature in train_data.columns:
            le = LabelEncoder()
            train_data[feature] = le.fit_transform(train_data[feature].astype(str))
            label_encoders[feature] = le
    
    # 4. 创建新特征
    print("创建新特征...")
    
    # 计算车龄（直接从日期计算）
    if 'regDate' in train_data.columns and 'creatDate' in train_data.columns:
        # 从注册日期提取年份
        regYear = train_data['regDate'] // 10000
        # 从创建日期提取年份
        creatYear = train_data['creatDate'] // 10000
        # 计算车龄
        train_data['car_age'] = creatYear - regYear
        # 处理异常车龄
        train_data.loc[train_data['car_age'] < 0, 'car_age'] = 0
        train_data.loc[train_data['car_age'] > 50, 'car_age'] = 50
    
    # 创建是否新车特征（购买年限小于1年的是新车）
    if 'car_age' in train_data.columns:
        train_data['is_new_car'] = (train_data['car_age'] < 1).astype(int)
        print(f"新车数量: {train_data['is_new_car'].sum()}, 占比: {train_data['is_new_car'].mean():.3f}")
    
    # 创建价格区间特征（提高泛化能力）
    if 'price' in train_data.columns:
        # 使用分位数创建价格区间
        price_quantiles = train_data['price'].quantile([0.25, 0.5, 0.75])
        train_data['price_level'] = pd.cut(train_data['price'], 
                                          bins=[0, price_quantiles[0.25], price_quantiles[0.5], 
                                                price_quantiles[0.75], train_data['price'].max()],
                                          labels=[0, 1, 2, 3], include_lowest=True)
        train_data['price_level'] = train_data['price_level'].astype(int)
    
    # 创建交互特征
    if 'power' in train_data.columns and 'car_age' in train_data.columns:
        train_data['power_per_age'] = train_data['power'] / (train_data['car_age'] + 1)
    
    if 'kilometer' in train_data.columns and 'car_age' in train_data.columns:
        train_data['km_per_age'] = train_data['kilometer'] / (train_data['car_age'] + 1)
    
    # 对价格进行对数变换以减少偏度
    if 'price' in train_data.columns:
        train_data['log_price'] = np.log1p(train_data['price'])
    
    print(f"预处理后数据形状: {train_data.shape}")
    
    return train_data, label_encoders

def prepare_features(data, label_encoders=None, is_training=True):
    """准备特征数据"""
    # 选择特征列
    feature_columns = [
        'gearbox', 'power', 'kilometer',
        'notRepairedDamage', 'model', 'bodyType', 'fuelType'
    ]
    
    # 添加v_特征
    v_features = [f'v_{i}' for i in range(15)]
    feature_columns.extend(v_features)
    
    # 添加brand的One-Hot编码特征
    if label_encoders and 'brand_columns' in label_encoders:
        brand_features = label_encoders['brand_columns']
        for feature in brand_features:
            if feature in data.columns:
                feature_columns.append(feature)
    
    # 添加新创建的特征
    new_features = ['car_age', 'is_new_car']
    for feature in new_features:
        if feature in data.columns:
            feature_columns.append(feature)
    
    # 添加交互特征
    interaction_features = ['power_per_age', 'km_per_age']
    for feature in interaction_features:
        if feature in data.columns:
            feature_columns.append(feature)
    
    # 确保所有特征都存在
    available_features = [col for col in feature_columns if col in data.columns]

    print(f"特征列表: {available_features}")
    
    if not is_training and label_encoders:
        # 对测试数据进行编码
        categorical_features = ['model', 'brand', 'bodyType', 'fuelType', 'gearbox', 'notRepairedDamage']
        
        # 对brand进行One-Hot编码
        if 'brand' in data.columns and 'brand_columns' in label_encoders:
            print("对测试数据的brand进行One-Hot编码...")
            brand_dummies = pd.get_dummies(data['brand'], prefix='brand')
            
            # 确保测试集具有和训练集相同的brand列
            train_brand_cols = label_encoders['brand_columns']
            for col in train_brand_cols:
                if col not in brand_dummies.columns:
                    brand_dummies[col] = 0
            
            # 只保留训练时存在的brand列
            brand_dummies = brand_dummies[train_brand_cols]
            
            # 将One-Hot编码的列添加到数据中
            data = pd.concat([data, brand_dummies], axis=1)
            print(f"测试集brand特征展开为 {brand_dummies.shape[1]} 个特征")
        
        # 对其他分类特征使用LabelEncoder
        other_categorical_features = [f for f in categorical_features if f != 'brand']
        for feature in other_categorical_features:
            if feature in data.columns and feature in label_encoders:
                # 处理训练时未见过的类别
                unique_values = set(data[feature].astype(str).unique())
                known_values = set(label_encoders[feature].classes_)
                new_values = unique_values - known_values
                
                if new_values:
                    print(f"发现 {feature} 中的新类别: {new_values}")
                    # 用最频繁的类别替换新类别
                    most_common = label_encoders[feature].classes_[0]
                    data[feature] = data[feature].astype(str).replace(list(new_values), most_common)
                
                data[feature] = label_encoders[feature].transform(data[feature].astype(str))
    
    X = data[available_features]
    print(f"使用的特征数量: {len(available_features)}")
    print(f"特征列表: {available_features}")
    
    return X

def build_neural_network(input_dim):
    """构建3层神经网络模型"""
    print("构建3层神经网络模型...")
    
    model = keras.Sequential([
        # 输入层
        layers.Input(shape=(input_dim,)),
        
        # 第1层：全连接层 + BatchNormalization + Dropout
        layers.Dense(512, activation='relu', name='dense_1'),
        # layers.BatchNormalization(name='bn_1'),
        # layers.Dropout(0.3, name='dropout_1'),
        
        # 第2层：全连接层 + BatchNormalization + Dropout
        layers.Dense(128, activation='relu', name='dense_2'),
        # layers.BatchNormalization(name='bn_2'),
        # layers.Dropout(0.2, name='dropout_2'),
        
        # 第3层（输出层）：单个神经元用于回归
        layers.Dense(1, activation='linear', name='output')
    ])
    
    print("\n网络架构:")
    model.summary()
    
    return model

def train_neural_network():
    """训练神经网络模型"""
    print("=" * 60)
    print("开始训练3层神经网络模型")
    print("=" * 60)
    
    # 加载和预处理数据
    train_data, label_encoders = load_and_preprocess_data()
    
    # 准备特征和目标变量
    X = prepare_features(train_data, label_encoders, is_training=True)
    y = train_data['price']  # 使用原始价格，不用对数变换
    
    print(f"特征数据形状: {X.shape}")
    print(f"目标变量形状: {y.shape}")
    
    # 划分训练集和验证集（使用分层采样）
    # 基于价格区间进行分层采样，确保各价格区间的分布一致
    if 'price_level' in train_data.columns:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, 
            stratify=train_data['price_level']
        )
    else:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=None
        )
    
    print(f"训练集大小: {X_train.shape[0]}")
    print(f"验证集大小: {X_val.shape[0]}")
    
    # 检查价格分布
    print(f"\n价格分布统计:")
    print(f"训练集价格 - 均值: {y_train.mean():.2f}, 标准差: {y_train.std():.2f}")
    print(f"验证集价格 - 均值: {y_val.mean():.2f}, 标准差: {y_val.std():.2f}")
    print(f"训练集价格 - 最小值: {y_train.min():.2f}, 最大值: {y_train.max():.2f}")
    print(f"验证集价格 - 最小值: {y_val.min():.2f}, 最大值: {y_val.max():.2f}")
    
    # 特征标准化（对神经网络很重要）
    print("特征标准化...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # 目标变量标准化（可选，通常有助于训练稳定性）
    # 注意：如果验证集损失异常低，可能需要关闭目标变量标准化
    use_target_scaling = True  # 可以设置为False来关闭目标变量标准化
    
    if use_target_scaling:
        y_scaler = StandardScaler()
        y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1)).flatten()
        y_val_scaled = y_scaler.transform(y_val.values.reshape(-1, 1)).flatten()
        print("使用目标变量标准化")
    else:
        y_scaler = None
        y_train_scaled = y_train.values
        y_val_scaled = y_val.values
        print("不使用目标变量标准化")
    
    # 构建模型
    model = build_neural_network(X_train_scaled.shape[1])
    
    # 编译模型
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mae',  # 使用MAE作为损失函数，与评估指标一致
        metrics=['mae', 'mse']
    )
    
    # 设置回调函数
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_mae',  # 监控MAE而不是loss
            patience=40,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_mae',  # 监控MAE而不是loss
            factor=0.5,
            patience=10,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    # 训练模型
    print("\n开始训练神经网络...")
    history = model.fit(
        X_train_scaled, y_train_scaled,
        batch_size=256,
        epochs=200,
        validation_data=(X_val_scaled, y_val_scaled),
        callbacks=callbacks,
        verbose=1
    )
    
    # 预测（还原标准化）
    y_train_pred_scaled = model.predict(X_train_scaled, verbose=0)
    y_val_pred_scaled = model.predict(X_val_scaled, verbose=0)
    
    # 还原标准化
    if y_scaler is not None:
        y_train_pred = y_scaler.inverse_transform(y_train_pred_scaled).flatten()
        y_val_pred = y_scaler.inverse_transform(y_val_pred_scaled).flatten()
    else:
        y_train_pred = y_train_pred_scaled.flatten()
        y_val_pred = y_val_pred_scaled.flatten()
    
    # 评估模型
    train_mse = mean_squared_error(y_train, y_train_pred)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    
    val_mse = mean_squared_error(y_val, y_val_pred)
    val_mae = mean_absolute_error(y_val, y_val_pred)
    val_r2 = r2_score(y_val, y_val_pred)
    
    print(f"\n训练集表现:")
    print(f"MSE: {train_mse:.2f}")
    print(f"MAE: {train_mae:.2f}")
    print(f"R²: {train_r2:.4f}")
    
    print(f"\n验证集表现:")
    print(f"MSE: {val_mse:.2f}")
    print(f"MAE: {val_mae:.2f}")
    print(f"R²: {val_r2:.4f}")
    
    # 绘制训练历史
    plot_training_history(history)
    
    return model, scaler, y_scaler, label_encoders, X.columns, val_mae

def plot_training_history(history):
    """绘制训练历史"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # 损失曲线
    ax1.plot(history.history['loss'], label='训练损失 / Training Loss')
    ax1.plot(history.history['val_loss'], label='验证损失 / Validation Loss')
    ax1.set_title('模型损失 / Model Loss')
    ax1.set_xlabel('轮次 / Epoch')
    ax1.set_ylabel('损失 / Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # MAE曲线
    ax2.plot(history.history['mae'], label='训练MAE / Training MAE')
    ax2.plot(history.history['val_mae'], label='验证MAE / Validation MAE')
    ax2.set_title('模型MAE / Model MAE')
    ax2.set_xlabel('轮次 / Epoch')
    ax2.set_ylabel('MAE')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('neural_network_training_history.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("训练历史图已保存为 neural_network_training_history.png")

def predict_test_set(model, scaler, y_scaler, label_encoders, feature_columns):
    """对测试集进行预测"""
    print("\n" + "=" * 60)
    print("开始预测测试集")
    print("=" * 60)
    
    # 加载测试数据
    print("加载测试数据...")
    test_data = pd.read_csv('used_car_testB_20200421.csv', sep=' ')
    
    print(f"测试数据形状: {test_data.shape}")
    print(f"测试数据列名: {test_data.columns.tolist()}")
    
    # 预处理测试数据（与训练数据相同的步骤）
    print("预处理测试数据...")
    
    # 1. 处理缺失值
    categorical_features = ['model', 'brand', 'bodyType', 'fuelType', 'gearbox', 'notRepairedDamage']
    for feature in categorical_features:
        if feature in test_data.columns:
            mode_value = test_data[feature].mode()[0] if not test_data[feature].mode().empty else 0
            test_data[feature].fillna(mode_value, inplace=True)
    
    numeric_features = ['power', 'kilometer'] + [f'v_{i}' for i in range(15)]
    for feature in numeric_features:
        if feature in test_data.columns:
            median_value = test_data[feature].median()
            test_data[feature].fillna(median_value, inplace=True)
    
    # 2. 处理异常值
    if 'power' in test_data.columns:
        test_data['power'] = pd.to_numeric(test_data['power'], errors='coerce')
        test_data['power'].fillna(test_data['power'].median(), inplace=True)
        test_data.loc[test_data['power'] > 1000, 'power'] = test_data['power'].median()
        test_data.loc[test_data['power'] < 0, 'power'] = test_data['power'].median()
    
    if 'kilometer' in test_data.columns:
        test_data['kilometer'] = pd.to_numeric(test_data['kilometer'], errors='coerce')
        test_data['kilometer'].fillna(test_data['kilometer'].median(), inplace=True)
        test_data.loc[test_data['kilometer'] > 1000000, 'kilometer'] = test_data['kilometer'].median()
        test_data.loc[test_data['kilometer'] < 0, 'kilometer'] = test_data['kilometer'].median()
    
    # 3. 创建新特征
    # 计算车龄（直接从日期计算）
    if 'regDate' in test_data.columns and 'creatDate' in test_data.columns:
        # 从注册日期提取年份
        regYear = test_data['regDate'] // 10000
        # 从创建日期提取年份
        creatYear = test_data['creatDate'] // 10000
        # 计算车龄
        test_data['car_age'] = creatYear - regYear
        # 处理异常车龄
        test_data.loc[test_data['car_age'] < 0, 'car_age'] = 0
        test_data.loc[test_data['car_age'] > 50, 'car_age'] = 50
    
    # 创建是否新车特征（购买年限小于1年的是新车）
    if 'car_age' in test_data.columns:
        test_data['is_new_car'] = (test_data['car_age'] < 1).astype(int)
        print(f"测试集新车数量: {test_data['is_new_car'].sum()}, 占比: {test_data['is_new_car'].mean():.3f}")
    
    # 创建交互特征
    if 'power' in test_data.columns and 'car_age' in test_data.columns:
        test_data['power_per_age'] = test_data['power'] / (test_data['car_age'] + 1)
    
    if 'kilometer' in test_data.columns and 'car_age' in test_data.columns:
        test_data['km_per_age'] = test_data['kilometer'] / (test_data['car_age'] + 1)
    
    # 4. 准备特征
    X_test = prepare_features(test_data, label_encoders, is_training=False)
    
    # 确保测试集具有和训练集相同的特征
    missing_features = set(feature_columns) - set(X_test.columns)
    if missing_features:
        print(f"测试集缺少特征: {missing_features}")
        for feature in missing_features:
            X_test[feature] = 0
    
    # 重新排序特征列
    X_test = X_test[feature_columns]
    
    print(f"测试集特征数据形状: {X_test.shape}")
    
    # 标准化
    X_test_scaled = scaler.transform(X_test)
    
    # 预测
    print("进行预测...")
    y_pred_scaled = model.predict(X_test_scaled, verbose=1)
    
    # 还原标准化
    y_pred = y_scaler.inverse_transform(y_pred_scaled).flatten()
    
    # 确保预测值为正数
    y_pred = np.maximum(y_pred, 0)
    
    # 创建结果DataFrame
    result = pd.DataFrame({
        'SaleID': test_data['SaleID'],
        'price': y_pred
    })
    
    print(f"预测结果统计:")
    print(f"预测价格数量: {len(y_pred)}")
    print(f"预测价格范围: {y_pred.min():.2f} - {y_pred.max():.2f}")
    print(f"预测价格均值: {y_pred.mean():.2f}")
    print(f"预测价格中位数: {np.median(y_pred):.2f}")
    
    # 保存结果
    output_file = 'neural_network_price_predictions_backup.csv'
    result.to_csv(output_file, index=False)
    print(f"\n预测结果已保存到: {output_file}")
    
    return result

def analyze_price_ranges(predictions):
    """分析预测结果的价格分布"""
    print("\n价格区间分布分析:")
    ranges = [
        (0, 2000, "0-2K"),
        (2000, 5000, "2K-5K"),
        (5000, 10000, "5K-10K"),
        (10000, 20000, "10K-20K"),
        (20000, float('inf'), "20K+")
    ]
    
    print("-" * 50)
    print(f"{'价格区间':<10} {'样本数':<8} {'百分比':<10}")
    print("-" * 50)
    
    for min_val, max_val, label in ranges:
        count = ((predictions['price'] >= min_val) & (predictions['price'] < max_val)).sum()
        percent = (count / len(predictions)) * 100
        print(f"{label:<10} {count:<8} {percent:<10.2f}%")

def main():
    """主函数"""
    print("开始3层神经网络二手车价格预测（备份版）")
    print("=" * 60)
    
    # 训练模型
    model, scaler, y_scaler, label_encoders, feature_columns, val_mae = train_neural_network()
    
    # 预测测试集
    predictions = predict_test_set(model, scaler, y_scaler, label_encoders, feature_columns)
    
    # 分析预测结果
    analyze_price_ranges(predictions)
    
    print("\n" + "=" * 60)
    print("神经网络模型训练和预测完成！")
    print(f"验证集MAE: {val_mae:.2f}元")
    print("=" * 60)
    
    # 显示前几个预测结果
    print("\n前10个预测结果:")
    print(predictions.head(10))
    
    return predictions

if __name__ == "__main__":
    predictions = main() 