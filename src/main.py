import warnings

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import process_one as po
import MedicalDataLoader as mdl


def process_all(dataset_name_list, data_path_list, data_load_fn_list, algorithm_list, process_fn_list, min_max_flag):
    for info in list(zip(dataset_name_list, data_path_list, data_load_fn_list)):
        dataset_name = info[0]
        data_path = info[1]
        data_load_fn = info[2]

        if 'lung' not in dataset_name:
            continue

        flag = False
        if min_max_flag and (dataset_name in ['breast-cancer', 'diabetes_2', 'sepsis']):
            flag = True

        print(f'------ {dataset_name} ------')

        X, y = data_load_fn(file_path=data_path)  # ndarray

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            stratify=y,
            random_state=42
        )

        # 数据 Min-Max 预处理
        if flag:
            scaler = MinMaxScaler(feature_range=(0, 1))
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        data_dict = {'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test}

        date = '0617'  # TODO
        doc_out_path = f'../output/{dataset_name}-result-{date}.txt'
        excel_out_path = f'../output/{dataset_name}-result-{date}.xlsx'
        if flag:
            doc_out_path = f'../output/{dataset_name}-result-MinMax-{date}.txt'
            excel_out_path = f'../output/{dataset_name}-result-MinMax-{date}.xlsx'

        doc_out = open(doc_out_path, 'w')

        po.process_one_data(
            dataset_name=dataset_name,
            data_dict=data_dict,
            doc_out_file=doc_out,
            excel_out_path=excel_out_path,
            algorithm_list=algorithm_list,
            process_fn_list=process_fn_list,
            min_max_flag=flag
        )

        doc_out.close()


if __name__ == '__main__':
    warnings.filterwarnings('ignore')

    names = [
        'breast-cancer',
        'cervical-cancer',
        'diabetes_risk',
        'sepsis'
    ]
    path_list = [
        '../datasets/breast-cancer-data.csv',
        '../datasets/cervical-cancer.csv',
        '../datasets/diabetes_risk_prediction_dataset.csv',
        '../datasets/sepsis.csv'
    ]
    data_load_fn_list = [
        mdl.load_breast_cancer,
        mdl.load_cervical_cancer,
        mdl.load_diabetes_risk,
        mdl.load_sepsis
    ]

    algorithm_list = ['KAN', 'MLP', 'XGB']
    process_fn_list = [
        po.process_one_kan,
        po.process_one_mlp,
        po.process_one_xgb
    ]

    min_max_flag = True
    process_all(
        dataset_name_list=names,
        data_path_list=path_list,
        data_load_fn_list=data_load_fn_list,
        algorithm_list=algorithm_list,
        process_fn_list=process_fn_list,
        min_max_flag=min_max_flag
    )




