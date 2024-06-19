import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder, StandardScaler


from preprocessing import preprocess_data


# 读取数据集：乳腺癌 breast-cancer-data.csv
# diagnosis为y，转换M和B数据
def load_breast_cancer(file_path):
    data = pd.read_csv(file_path)
    yname = 'diagnosis'  # 'M' -> 1, 'B' -> 0
    data[yname] = list(map(lambda x: 1 if x == 'M' else 0, data[yname]))  # binary
    X = data.drop([yname], axis=1)
    y = data[yname]
    X = X.values  # ndarray
    y = y.values  # ndarray
    return X, y


# 读取数据集：cardio.csv
def load_cardio(file_path):
    data = pd.read_csv(file_path, sep=';')
    data['age'] = (data['age'] / 365).astype('int')
    data.drop('id', axis=1, inplace=True)
    cols = data.columns
    y_cols = cols[-1]
    X = data.drop(y_cols, axis=1)
    y = data.cardio
    X = X.values  # ndarray
    y = y.values  # ndarray
    return X, y


# 读取数据集：宫颈癌 cervical-cancer.csv
def load_cervical_cancer(file_path):
    data = pd.read_csv(file_path)
    yname = 'Dx:Cancer'
    X = data.drop([yname], axis=1).fillna(0)
    y = data[yname]

    X = X.values  # ndarray
    y = y.values  # ndarray
    return X, y


# 读取数据集：糖尿病-2 diabetes_2.csv
def load_diabetes(file_path):
    data = pd.read_csv(file_path)
    y_cols = data.columns[-1]
    X = data.drop(y_cols, axis=1)
    y = data.Outcome
    X = X.values  # ndarray
    y = y.values  # ndarray
    return X, y


# 读取数据集：糖尿病风险预测 diabetes_risk_prediction_dataset.csv
def load_diabetes_risk(file_path):
    df = pd.read_csv(file_path)

    # Encode the target variable
    encode = LabelEncoder()
    df['class'] = encode.fit_transform(df['class'])

    # Fill missing values for numerical columns with the mean
    df.fillna(df.select_dtypes(include=[np.number]).mean(), inplace=True)

    y_cols = df.columns[-1]
    X = df.drop(y_cols, axis=1)
    y = df[y_cols]

    X_encoded = pd.get_dummies(X)
    X_encoded = X_encoded.astype(np.float32)

    X_encoded = X_encoded.values  # ndarray
    y = y.values  # ndarray

    return X_encoded, y


# 读取数据集：脓毒症 sepsis.csv  Paitients_Files_Train.csv
def load_sepsis(file_path):
    data = pd.read_csv(file_path)
    data['Sepsis'] = list(map(lambda x: 1 if x == 'Positive' else 0, data.Sepssis))  # binary
    X = data.drop(['ID', 'Sepssis', 'Sepsis', 'Insurance'], axis=1)
    y = data.Sepsis
    X = X.values  # ndarray
    y = y.values  # ndarray
    return X, y


# 读取数据集：肺癌 survey lung cancer.csv
def load_lung_cancer(file_path):
    data = pd.read_csv(file_path)
    cols = data.columns
    y_cols = cols[-1]
    data[y_cols] = 2 - data[y_cols]  # 原始：YES=1, NO=2
    X = data.drop(y_cols, axis=1)
    y = data[y_cols]
    X = X.values  # ndarray
    y = y.values  # ndarray
    return X, y


def load_patient_data(file_path):
    data = pd.read_csv(file_path)
    cols = data.columns
    df = preprocess_data(data)
    X = df.drop(columns='readmit72')
    y = df['readmit72']
    # Identify indices of all positive and negative samples
    positive_indices = np.where(y.values == 1)[0]
    negative_indices = np.where(y.values == 0)[0]

    # Sample negative indices to match the number of positive samples
    sampled_negative_indices = np.random.choice(negative_indices, size=len(positive_indices), replace=True)

    # Combine and shuffle indices
    sampled_indices = np.concatenate([positive_indices, sampled_negative_indices])
    np.random.shuffle(sampled_indices)

    # Select sampled data from the original dataset
    X = X.values[sampled_indices]
    y = y.values[sampled_indices]

    return X, y
