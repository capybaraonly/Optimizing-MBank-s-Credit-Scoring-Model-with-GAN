import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import warnings
warnings.filterwarnings('ignore')

# 加载数据
train_origin = pd.read_csv('./train.csv')
test_origin = pd.read_csv('./testA.csv')

# 抽取 10% 的数据
train = train_origin.sample(frac=0.1, random_state=42)
test = test_origin.sample(frac=0.1, random_state=42)

#查看前五行和后五行
# print(train.head(5).append(train.tail(5)))

# 数据质量分析
# 查看缺省值
d = (train.isnull().sum()/len(train)).to_dict()
# print(d)
# print('*'*30)
# print(f'There are {train.isnull().any().sum()} columns in train trainset with missing values.')
# 可视化
# missing_plt = train.isnull().sum()/len(train)
# missing_plt.plot.bar(figsize = (20,6))
# plt.title('Feature with Missing Value')
# plt.xlabel('Feature')
# plt.ylabel('Missing Number')
# plt.tight_layout()  # 调整布局，确保图形不被截断
# plt.show()

# 查看是否有只有一个值的字段
one_value_fea = [col for col in train.columns if train[col].nunique() <= 1]
one_value_fea_test = [col for col in test.columns if test[col].nunique() <= 1]
# print(one_value_fea)
# print('---------')
# print(one_value_fea_test)

# 违约与非违约样本数
class_counts = train.isDefault.value_counts()
# print("\n违约与非违约样本数:")
# print(class_counts)
# print(train.isDefault.value_counts(normalize=True))

# 区分数值型特征和非数值型特征
numerical_fea = train.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_fea = train.select_dtypes(exclude=['int64', 'float64']).columns.tolist()

# 输出结果
# print("数值型特征：", numerical_fea)
# print("非数值型特征：", categorical_fea)

#统计数值型类别的特征（连续型和离散型）
def get_numerical_continus_fea(data,feas):
    numerical_continus_fea = []
    numerical_discrete_fea = []
    for fea in feas:
        temp = data[fea].nunique()
        # 自定义变量的值的取值个数小于10为离散型变量
        if temp <= 10:
            numerical_discrete_fea.append(fea)
            continue
        numerical_continus_fea.append(fea)
    return numerical_continus_fea,numerical_discrete_fea
numerical_continus_fea,numerical_discrete_fea = get_numerical_continus_fea(train,numerical_fea)

# print('连续型数值特征:', numerical_continus_fea)
# print('离散型数值特征:', numerical_discrete_fea)

# 离散性数值类型
homeOwnership_counts = train['homeOwnership'].value_counts(dropna=False)
employmentLength_counts = train['employmentLength'].value_counts(dropna=False)
# # 设置图形风格和大小
# plt.style.use('bmh')
# plt.figure(figsize=(20, 6))
# # 第一个子图：homeOwnership
# plt.subplot(1, 2, 1)
# sns.barplot(x=homeOwnership_counts.keys(), y=homeOwnership_counts.values)
# plt.title('Feature homeOwnership Value Overview')
# plt.xlabel('Home Ownership')
# plt.ylabel('Count')
# # 第二个子图：employmentLength
# plt.subplot(1, 2, 2)
# # 只显示前20个值
# sns.barplot(x=employmentLength_counts.values[:20], y=employmentLength_counts.keys()[:20])
# plt.title('Feature employmentLength Value Overview')
# plt.xlabel('Employment Length')
# plt.ylabel('Count')
# # 显示图形
# plt.tight_layout()  # 调整布局
# plt.show()

# train_sample = train.sample(frac=0.1)  # 抽取10%的数据
# # 数值连续型特征概率分布可视化//图没跑
# f = pd.melt(train_sample, value_vars=numerical_continus_fea)
# g = sns.FacetGrid(f, col="variable", col_wrap=4, sharex=False, sharey=False)
# g = g.map(sns.histplot, "value", kde=True)  
# plt.show()

# loanAmount Values Distribution and after log
# plt.figure(figsize=(20, 12))

# plt.suptitle('loanAmount Values Distribution', fontsize=22)
# plt.subplot(221)
# sub_plot_1 = sns.histplot(train_sample['loanAmnt'], kde=True)
# sub_plot_1.set_title("loanAmnt Distribution", fontsize=18)
# sub_plot_1.set_xlabel("")
# sub_plot_1.set_ylabel("Probability", fontsize=15)

# plt.subplot(222)
# sub_plot_2 = sns.histplot(np.log(train_sample['loanAmnt']), kde=True)
# sub_plot_2.set_title("loanAmnt (Log) Distribution", fontsize=18)
# sub_plot_2.set_xlabel("")
# sub_plot_2.set_ylabel("Probability", fontsize=15)

# # 显示图形
# plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # 调整布局，为标题留出空间
# plt.show()

# # 单特征箱线图，区分是否违约
# box_fea = [ 'loanAmnt', 'interestRate', 'installment',  'postCode', 'regionCode', 'openAcc', 'totalAcc', 'n2', 'n3']
# f, ax = plt.subplots(3,3, figsize = (20,15))

# for i, col in enumerate(box_fea):
#     sns.boxplot(x = 'isDefault', y = col, saturation=0.5, palette='pastel', data = train, ax = ax[i//3][i%3])

# # 类别型变量X和y
# train_loan_isDefault = train.loc[train['isDefault'] == 1]
# train_loan_nonDefault= train.loc[train['isDefault'] == 0]

# fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 10))
# train_loan_isDefault.groupby('grade')['grade'].count().plot(kind='barh', ax=ax1, title='Count of grade Default')
# train_loan_nonDefault.groupby('grade')['grade'].count().plot(kind='barh', ax=ax2, title='Count of grade non-Default')
# train_loan_isDefault.groupby('employmentLength')['employmentLength'].count().plot(kind='barh', ax=ax3, title='Count of employmentLength Default')
# train_loan_nonDefault.groupby('employmentLength')['employmentLength'].count().plot(kind='barh', ax=ax4, title='Count of employmentLength non-Default')
# plt.tight_layout()
# plt.show()




#2、特征工程
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from catboost import CatBoostRegressor
import warnings
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, log_loss
warnings.filterwarnings('ignore')

#缺省值查看
numerical_fea = list(train.select_dtypes(exclude=['object']).columns)
category_fea = list(filter(lambda x: x not in numerical_fea,list(train.columns)))
label = 'isDefault'
numerical_fea.remove(label)
# print(train.isnull().sum())

#按照平均数填充数值型特征
train[numerical_fea] = train[numerical_fea].fillna(train[numerical_fea].median())
test[numerical_fea] = test[numerical_fea].fillna(train[numerical_fea].median())

#按照众数填充类别型特征
train[category_fea] = train[category_fea].fillna(train[category_fea].mode())
test[category_fea] = test[category_fea].fillna(train[category_fea].mode())

# print(train.isnull().sum())


# 其中的issue为时间格式，需要转化
# 转化成时间格式
for data in [train, test]:
    data['issueDate'] = pd.to_datetime(data['issueDate'],format='%Y-%m-%d')
    startdate = datetime.datetime.strptime('2007-06-01', '%Y-%m-%d')
    #构造时间特征
    data['issueDateDT'] = data['issueDate'].apply(lambda x: x-startdate).dt.days

# 处理 earliesCreditLine 列
time_columns = ['earliesCreditLine'] 
for col in time_columns:
    # 将时间格式的字符串转换为 datetime 对象
    train[col] = pd.to_datetime(train[col], format='%Y/%m/%d', errors='coerce')
    test[col] = pd.to_datetime(test[col], format='%Y/%m/%d', errors='coerce')
    
    # 计算与基准日期的天数差
    startdate = datetime.datetime.strptime('1990-01-01', '%Y-%m-%d')
    train[col + 'DT'] = train[col].apply(lambda x: (x - startdate).days if not pd.isna(x) else 0)
    test[col + 'DT'] = test[col].apply(lambda x: (x - startdate).days if not pd.isna(x) else 0)
    
    # 移除原始时间列
    train = train.drop(columns=[col])
    test = test.drop(columns=[col])

# 下面在对类别型的数据进行编码，即映射成数值型数据，以便于后期的模型使用
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le = LabelEncoder()
train['grade'] = le.fit_transform(train['grade'])
train['subGrade'] = le.fit_transform(train['subGrade'])
train['employmentLength'] = train['employmentLength'].apply(lambda x : str(x))
train['employmentLength'] = le.fit_transform(train['employmentLength'])
test['grade'] = le.fit_transform(test['grade'])
test['subGrade'] = le.fit_transform(test['subGrade'])
test['employmentLength'] = test['employmentLength'].apply(lambda x : str(x))
test['employmentLength'] = le.fit_transform(data['employmentLength'])


import os
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, log_loss, classification_report, confusion_matrix
# import tensorflow as tf
from keras.api._v2.keras.models import Sequential
from keras.api._v2.keras.layers import Dense, Dropout
from keras.api._v2.keras.optimizers import Adam
from keras.api._v2.keras.utils import to_categorical
from imblearn.over_sampling import SMOTE
# 设置TensorFlow日志级别
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0: INFO, 1: WARNING, 2: ERROR, 3: FATAL

# 分割特征和标签
X = train.drop(columns=['isDefault', 'issueDate'])  # 确保移除 issueDate 列
y = train['isDefault']

# 数据集划分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 特征标准化
scaler = StandardScaler()
X_train[numerical_fea] = scaler.fit_transform(X_train[numerical_fea])
X_test[numerical_fea] = scaler.transform(X_test[numerical_fea])

# 将数据转换为 numpy 数组
X_train = X_train.values
X_test = X_test.values
y_train = y_train.values
y_test = y_test.values

# 生成对抗网络（GAN）模型
# 定义生成器
def build_generator(input_dim, output_dim):
    model = Sequential()
    model.add(Dense(256, input_dim=input_dim, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(output_dim, activation='linear'))
    return model

# 定义判别器
def build_discriminator(input_dim):
    model = Sequential()
    model.add(Dense(256, input_dim=input_dim, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    return model

# 构建 GAN
def build_gan(generator, discriminator):
    model = Sequential()
    model.add(generator)
    model.add(discriminator)
    return model

# 参数设置
input_dim = X_train.shape[1]
output_dim = X_train.shape[1]

# 构建生成器和判别器
generator = build_generator(input_dim, output_dim)
discriminator = build_discriminator(input_dim)
discriminator.compile(loss='binary_crossentropy', optimizer=Adam(0.0001, 0.5), metrics=['accuracy'])

# 构建 GAN
gan = build_gan(generator, discriminator)
gan.compile(loss='binary_crossentropy', optimizer=Adam(0.0001, 0.5))

# 训练 GAN
def train_gan(generator, discriminator, gan, X_train, y_train, epochs=10000, batch_size=64, sample_interval=1000):
    # 分离少数类和多数类
    X_minority = X_train[y_train == 1]
    X_majority = X_train[y_train == 0]

    # 标签
    valid = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))

    for epoch in range(epochs):
        # 训练判别器
        # 真实样本
        idx = np.random.randint(0, X_minority.shape[0], batch_size)
        real_samples = X_minority[idx]
        d_loss_real = discriminator.train_on_batch(real_samples, valid)

        # 生成样本
        noise = np.random.normal(0, 1, (batch_size, input_dim))
        gen_samples = generator.predict(noise)
        d_loss_fake = discriminator.train_on_batch(gen_samples, fake)

        # 总判别器损失
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # 训练生成器
        noise = np.random.normal(0, 1, (batch_size, input_dim))
        g_loss = gan.train_on_batch(noise, valid)

        # 打印进度
        if epoch % sample_interval == 0:
            print(f"Epoch: {epoch}, D Loss: {d_loss[0]}, D Acc: {100*d_loss[1]}, G Loss: {g_loss}")

    # 返回生成器和 X_minority
    return generator, X_minority

# 训练 GAN 并获取生成器和 X_minority
generator, X_minority = train_gan(generator, discriminator, gan, X_train, y_train, epochs=20000, batch_size=64, sample_interval=1000)

# 生成合成的少数类样本
noise = np.random.normal(0, 1, (X_minority.shape[0], input_dim))
synthetic_minority = generator.predict(noise)

# 合成数据集
X_train_gan = np.vstack([X_train, synthetic_minority])
y_train_gan = np.hstack([y_train, np.ones(synthetic_minority.shape[0])])

# 逻辑回归模型
lr = LogisticRegression(penalty='l2', C=1.0, solver='liblinear', max_iter=100, random_state=42, class_weight='balanced')
lr.fit(X_train_gan, y_train_gan)

# 预测
y_pred = lr.predict(X_test)
y_proba = lr.predict_proba(X_test)[:, 1]

# 评估
print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("ROC AUC Score:", roc_auc_score(y_test, y_proba))
print("Log Loss:", log_loss(y_test, y_proba))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
