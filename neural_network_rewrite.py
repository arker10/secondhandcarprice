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
    
    # 对bodyType进行One-Hot编码
    if 'bodyType' in train_data.columns:
        print("对bodyType进行One-Hot编码...")
        bodyType_dummies = pd.get_dummies(train_data['bodyType'], prefix='bodyType')
        print(f"bodyType特征展开为 {bodyType_dummies.shape[1]} 个特征")
        # 将One-Hot编码的列添加到数据中
        train_data = pd.concat([train_data, bodyType_dummies], axis=1)
        # 保存bodyType的唯一值供测试时使用
        label_encoders['bodyType_columns'] = bodyType_dummies.columns.tolist()
    
    # 对其他分类特征使用LabelEncoder
    other_categorical_features = [f for f in categorical_features if f != 'brand' and f != 'bodyType']
    for feature in other_categorical_features:
        if feature in train_data.columns:
            le = LabelEncoder()
            train_data[feature] = le.fit_transform(train_data[feature].astype(str))
            label_encoders[feature] = le
    
    # 4. 创建新特征
    print("创建新特征...")
    
    # 从注册日期提取年月日
    if 'regDate' in train_data.columns:
        train_data['regYear'] = train_data['regDate'] // 10000
        train_data['regMonth'] = (train_data['regDate'] % 10000) // 100
        train_data['regDay'] = train_data['regDate'] % 100
    
    # 从创建日期提取年月日
    if 'creatDate' in train_data.columns:
        train_data['creatYear'] = train_data['creatDate'] // 10000
        train_data['creatMonth'] = (train_data['creatDate'] % 10000) // 100
        train_data['creatDay'] = train_data['creatDate'] % 100
    
    # 计算车龄（使用年份差）
    if 'regYear' in train_data.columns and 'creatYear' in train_data.columns:
        train_data['car_age'] = train_data['creatYear'] - train_data['regYear']
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
    
    # 创建一些交互特征
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
    
    # 添加bodyType的One-Hot编码特征
    if label_encoders and 'bodyType_columns' in label_encoders:
        bodyType_features = label_encoders['bodyType_columns']
        for feature in bodyType_features:
            if feature in data.columns:
                feature_columns.append(feature)
    
    # 添加新创建的特征
    new_features = ['regYear', 'regMonth', 'regDay', 'creatYear', 'creatMonth', 'creatDay', 'car_age', 'is_new_car']
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
        
        # 对bodyType进行One-Hot编码
        if 'bodyType' in data.columns and 'bodyType_columns' in label_encoders:
            print("对测试数据的bodyType进行One-Hot编码...")
            bodyType_dummies = pd.get_dummies(data['bodyType'], prefix='bodyType')
            
            # 确保测试集具有和训练集相同的bodyType列
            train_bodyType_cols = label_encoders['bodyType_columns']
            for col in train_bodyType_cols:
                if col not in bodyType_dummies.columns:
                    bodyType_dummies[col] = 0
            
            # 只保留训练时存在的bodyType列
            bodyType_dummies = bodyType_dummies[train_bodyType_cols]
            
            # 将One-Hot编码的列添加到数据中
            data = pd.concat([data, bodyType_dummies], axis=1)
            print(f"测试集bodyType特征展开为 {bodyType_dummies.shape[1]} 个特征")
        
        # 对其他分类特征使用LabelEncoder
        other_categorical_features = [f for f in categorical_features if f != 'brand' and f != 'bodyType']
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
        layers.BatchNormalization(name='bn_1'),
        layers.Dropout(0.1, name='dropout_1'),
        
        # 第2层：全连接层 + BatchNormalization + Dropout
        layers.Dense(128, activation='relu', name='dense_2'),
        layers.BatchNormalization(name='bn_2'),
        layers.Dropout(0.1, name='dropout_2'),
        
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
    
    # 特征标准化（对神经网络很重要）
    print("特征标准化...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # 目标变量标准化（可选，通常有助于训练稳定性）
    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1)).flatten()
    y_val_scaled = y_scaler.transform(y_val.values.reshape(-1, 1)).flatten()
    
    # 构建模型
    model = build_neural_network(X_train_scaled.shape[1])
    
    # 编译模型
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0005),
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
            patience=20,
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
    y_train_pred = y_scaler.inverse_transform(y_train_pred_scaled).flatten()
    y_val_pred = y_scaler.inverse_transform(y_val_pred_scaled).flatten()
    
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
    
    # 分析预测值与验证集的差异
    analyze_prediction_vs_validation(y_train, y_train_pred, y_val, y_val_pred)
    
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

def analyze_prediction_vs_validation(y_train, y_train_pred, y_val, y_val_pred):
    """分析预测值与验证集的差异"""
    print("\n" + "=" * 60)
    print("预测值与验证集差异分析")
    print("=" * 60)
    
    # 添加调试信息
    print(f"数据范围检查:")
    print(f"训练集真实价格范围: {y_train.min():.2f} - {y_train.max():.2f}")
    print(f"训练集预测价格范围: {y_train_pred.min():.2f} - {y_train_pred.max():.2f}")
    print(f"验证集真实价格范围: {y_val.min():.2f} - {y_val.max():.2f}")
    print(f"验证集预测价格范围: {y_val_pred.min():.2f} - {y_val_pred.max():.2f}")
    
    # 计算各种误差指标
    train_errors = np.abs(y_train - y_train_pred)
    val_errors = np.abs(y_val - y_val_pred)
    
    print(f"训练集误差统计:")
    print(f"  绝对误差均值: {train_errors.mean():.2f}")
    print(f"  绝对误差中位数: {np.median(train_errors):.2f}")
    print(f"  绝对误差标准差: {train_errors.std():.2f}")
    print(f"  绝对误差最小值: {train_errors.min():.2f}")
    print(f"  绝对误差最大值: {train_errors.max():.2f}")
    
    print(f"\n验证集误差统计:")
    print(f"  绝对误差均值: {val_errors.mean():.2f}")
    print(f"  绝对误差中位数: {np.median(val_errors):.2f}")
    print(f"  绝对误差标准差: {val_errors.std():.2f}")
    print(f"  绝对误差最小值: {val_errors.min():.2f}")
    print(f"  绝对误差最大值: {val_errors.max():.2f}")
    
    # 计算相对误差（避免除零）
    train_valid_mask = y_train > 0
    val_valid_mask = y_val > 0
    
    if train_valid_mask.sum() > 0:
        train_relative_errors = (train_errors[train_valid_mask] / y_train[train_valid_mask]) * 100
        print(f"\n训练集相对误差统计:")
        print(f"  相对误差均值: {train_relative_errors.mean():.2f}%")
        print(f"  相对误差中位数: {np.median(train_relative_errors):.2f}%")
        print(f"  相对误差标准差: {train_relative_errors.std():.2f}%")
    
    if val_valid_mask.sum() > 0:
        val_relative_errors = (val_errors[val_valid_mask] / y_val[val_valid_mask]) * 100
        print(f"\n验证集相对误差统计:")
        print(f"  相对误差均值: {val_relative_errors.mean():.2f}%")
        print(f"  相对误差中位数: {np.median(val_relative_errors):.2f}%")
        print(f"  相对误差标准差: {val_relative_errors.std():.2f}%")
    
    # 按价格区间分析误差
    print(f"\n按价格区间的误差分析:")
    price_ranges = [
        (0, 5000, "0-5K"),
        (5000, 10000, "5K-10K"),
        (10000, 20000, "10K-20K"),
        (20000, 50000, "20K-50K"),
        (50000, float('inf'), "50K+")
    ]
    
    print("-" * 80)
    print(f"{'价格区间':<10} {'样本数':<8} {'真值均值':<10} {'预测均值':<10} {'MAE':<10} {'相对误差%':<10}")
    print("-" * 80)
    
    for min_val, max_val, label in price_ranges:
        # 训练集分析
        train_mask = (y_train >= min_val) & (y_train < max_val)
        if train_mask.sum() > 0:
            train_subset_true = y_train[train_mask]
            train_subset_pred = y_train_pred[train_mask]
            train_subset_mae = mean_absolute_error(train_subset_true, train_subset_pred)
            train_subset_relative = (train_subset_mae / train_subset_true.mean()) * 100
            
            print(f"训练-{label:<6} {train_mask.sum():<8} {train_subset_true.mean():<10.2f} {train_subset_pred.mean():<10.2f} {train_subset_mae:<10.2f} {train_subset_relative:<10.2f}")
        
        # 验证集分析
        val_mask = (y_val >= min_val) & (y_val < max_val)
        if val_mask.sum() > 0:
            val_subset_true = y_val[val_mask]
            val_subset_pred = y_val_pred[val_mask]
            val_subset_mae = mean_absolute_error(val_subset_true, val_subset_pred)
            val_subset_relative = (val_subset_mae / val_subset_true.mean()) * 100
            
            print(f"验证-{label:<6} {val_mask.sum():<8} {val_subset_true.mean():<10.2f} {val_subset_pred.mean():<10.2f} {val_subset_mae:<10.2f} {val_subset_relative:<10.2f}")
    
    # 分析各价格区间对最终MAE的影响
    print(f"\n价格区间对最终MAE的影响分析:")
    print("=" * 80)
    
    # 计算总体MAE
    total_mae = val_errors.mean()
    total_samples = len(y_val)
    
    # 存储各区间的影响信息
    interval_impacts = []
    
    for min_val, max_val, label in price_ranges:
        val_mask = (y_val >= min_val) & (y_val < max_val)
        if val_mask.sum() > 0:
            val_subset_true = y_val[val_mask]
            val_subset_pred = y_val_pred[val_mask]
            val_subset_mae = mean_absolute_error(val_subset_true, val_subset_pred)
            val_subset_errors = np.abs(val_subset_true - val_subset_pred)
            
            # 计算该区间对总体MAE的贡献
            interval_contribution = val_subset_errors.sum() / total_samples
            interval_weight = val_mask.sum() / total_samples
            interval_impact_score = interval_contribution / total_mae * 100  # 百分比影响
            
            interval_impacts.append({
                'label': label,
                'sample_count': val_mask.sum(),
                'weight': interval_weight,
                'mae': val_subset_mae,
                'contribution': interval_contribution,
                'impact_score': interval_impact_score,
                'avg_price': val_subset_true.mean(),
                'pred_avg_price': val_subset_pred.mean()
            })
    
    # 按影响大小排序
    interval_impacts.sort(key=lambda x: x['impact_score'], reverse=True)
    
    print(f"{'价格区间':<12} {'样本数':<8} {'权重%':<8} {'区间MAE':<10} {'贡献度':<10} {'影响%':<8} {'真值均值':<10} {'预测均值':<10}")
    print("-" * 80)
    
    for impact in interval_impacts:
        print(f"{impact['label']:<12} {impact['sample_count']:<8} {impact['weight']*100:<8.2f} {impact['mae']:<10.2f} {impact['contribution']:<10.2f} {impact['impact_score']:<8.2f} {impact['avg_price']:<10.2f} {impact['pred_avg_price']:<10.2f}")
    
    print("-" * 80)
    print(f"总体MAE: {total_mae:.2f}")
    print(f"总样本数: {total_samples}")
    
    # 找出影响最大的区间
    if interval_impacts:
        max_impact = interval_impacts[0]
        print(f"\n对最终MAE影响最大的价格区间: {max_impact['label']}")
        print(f"  影响程度: {max_impact['impact_score']:.2f}%")
        print(f"  样本权重: {max_impact['weight']*100:.2f}%")
        print(f"  区间MAE: {max_impact['mae']:.2f}")
        
        # 分析前3个影响最大的区间
        print(f"\n前3个影响最大的价格区间:")
        for i, impact in enumerate(interval_impacts[:3], 1):
            print(f"{i}. {impact['label']}: 影响{impact['impact_score']:.2f}% (权重{impact['weight']*100:.2f}%, MAE{impact['mae']:.2f})")
    
    # 分析样本分布对MAE的影响
    print(f"\n样本分布对MAE的影响分析:")
    high_weight_intervals = [imp for imp in interval_impacts if imp['weight'] > 0.1]  # 权重超过10%的区间
    if high_weight_intervals:
        print("高权重区间（权重>10%）:")
        for imp in high_weight_intervals:
            print(f"  {imp['label']}: 权重{imp['weight']*100:.2f}%, 贡献{imp['impact_score']:.2f}%")
    else:
        print("没有权重超过10%的区间")
    
    # 分析高MAE区间
    high_mae_intervals = [imp for imp in interval_impacts if imp['mae'] > total_mae * 1.5]  # MAE超过总体1.5倍的区间
    if high_mae_intervals:
        print("\n高MAE区间（MAE>总体1.5倍）:")
        for imp in high_mae_intervals:
            print(f"  {imp['label']}: MAE{imp['mae']:.2f} (总体{total_mae:.2f}的{imp['mae']/total_mae:.2f}倍)")
    else:
        print("\n没有MAE超过总体1.5倍的区间")
    
    # 分析过拟合情况
    print(f"\n过拟合分析:")
    print(f"训练集MAE: {train_errors.mean():.2f}")
    print(f"验证集MAE: {val_errors.mean():.2f}")
    mae_diff = train_errors.mean() - val_errors.mean()
    if mae_diff > 0:
        print(f"验证集MAE比训练集低 {mae_diff:.2f}，可能存在数据分布不一致问题")
    else:
        print(f"训练集MAE比验证集低 {abs(mae_diff):.2f}，可能存在过拟合")
    
    # 找出误差最大的样本
    print(f"\n验证集误差最大的10个样本:")
    val_error_df = pd.DataFrame({
        'true_price': y_val,
        'pred_price': y_val_pred,
        'absolute_error': val_errors,
        'percentage_error': (val_errors / y_val) * 100 if y_val.sum() > 0 else np.zeros(len(y_val))
    })
    
    worst_predictions = val_error_df.nlargest(10, 'absolute_error')
    print(worst_predictions.round(2))
    
    # 创建误差分析可视化
    create_error_analysis_plots(y_train, y_train_pred, y_val, y_val_pred, train_errors, val_errors)
    
    print("=" * 60)

def create_error_analysis_plots(y_train, y_train_pred, y_val, y_val_pred, train_errors, val_errors):
    """创建误差分析可视化图表"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('预测误差分析 / Prediction Error Analysis', fontsize=16, fontweight='bold')
    
    # 1. 训练集真实值 vs 预测值
    axes[0, 0].scatter(y_train, y_train_pred, alpha=0.5, s=1, color='blue')
    axes[0, 0].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
    axes[0, 0].set_xlabel('真实价格 / True Price')
    axes[0, 0].set_ylabel('预测价格 / Predicted Price')
    axes[0, 0].set_title('训练集: 真实值 vs 预测值\nTraining: True vs Predicted')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 验证集真实值 vs 预测值
    axes[0, 1].scatter(y_val, y_val_pred, alpha=0.5, s=1, color='green')
    axes[0, 1].plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--', lw=2)
    axes[0, 1].set_xlabel('真实价格 / True Price')
    axes[0, 1].set_ylabel('预测价格 / Predicted Price')
    axes[0, 1].set_title('验证集: 真实值 vs 预测值\nValidation: True vs Predicted')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 训练集误差分布
    axes[0, 2].hist(train_errors, bins=50, alpha=0.7, edgecolor='black', color='blue')
    axes[0, 2].axvline(train_errors.mean(), color='red', linestyle='--', linewidth=2, 
                       label=f'MAE: {train_errors.mean():.2f}')
    axes[0, 2].set_xlabel('绝对误差 / Absolute Error')
    axes[0, 2].set_ylabel('频次 / Frequency')
    axes[0, 2].set_title('训练集误差分布\nTraining Error Distribution')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. 验证集误差分布
    axes[1, 0].hist(val_errors, bins=50, alpha=0.7, edgecolor='black', color='green')
    axes[1, 0].axvline(val_errors.mean(), color='red', linestyle='--', linewidth=2, 
                       label=f'MAE: {val_errors.mean():.2f}')
    axes[1, 0].set_xlabel('绝对误差 / Absolute Error')
    axes[1, 0].set_ylabel('频次 / Frequency')
    axes[1, 0].set_title('验证集误差分布\nValidation Error Distribution')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. 误差 vs 真实价格（训练集）
    axes[1, 1].scatter(y_train, train_errors, alpha=0.5, s=1, color='blue')
    axes[1, 1].set_xlabel('真实价格 / True Price')
    axes[1, 1].set_ylabel('绝对误差 / Absolute Error')
    axes[1, 1].set_title('训练集: 误差 vs 真实价格\nTraining: Error vs True Price')
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. 误差 vs 真实价格（验证集）
    axes[1, 2].scatter(y_val, val_errors, alpha=0.5, s=1, color='green')
    axes[1, 2].set_xlabel('真实价格 / True Price')
    axes[1, 2].set_ylabel('绝对误差 / Absolute Error')
    axes[1, 2].set_title('验证集: 误差 vs 真实价格\nValidation: Error vs True Price')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('prediction_error_analysis_rewrite.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("误差分析图表已保存为 prediction_error_analysis_rewrite.png")

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
    # 从注册日期提取年月日
    if 'regDate' in test_data.columns:
        test_data['regYear'] = test_data['regDate'] // 10000
        test_data['regMonth'] = (test_data['regDate'] % 10000) // 100
        test_data['regDay'] = test_data['regDate'] % 100
    
    # 从创建日期提取年月日
    if 'creatDate' in test_data.columns:
        test_data['creatYear'] = test_data['creatDate'] // 10000
        test_data['creatMonth'] = (test_data['creatDate'] % 10000) // 100
        test_data['creatDay'] = test_data['creatDate'] % 100
    
    # 计算车龄（使用年份差）
    if 'regYear' in test_data.columns and 'creatYear' in test_data.columns:
        test_data['car_age'] = test_data['creatYear'] - test_data['regYear']
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
    output_file = 'neural_network_price_predictions_rewrite.csv'
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
    print("开始3层神经网络二手车价格预测（重写版）")
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