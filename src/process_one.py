import pandas as pd
import matplotlib.pyplot as plt
import torch
import lightning as L
from torch.utils.data import DataLoader, TensorDataset

from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV
from KAN import KAN
from MLP import MLP

from metric import evaluate
from metric import METRIC_INDEX_ORDER


def process_one_data(
        dataset_name, data_dict, doc_out_file, excel_out_path,
        algorithm_list, process_fn_list, min_max_flag
):
    result_dict = dict()  # 用于画图 ROC, PR
    rst_df = pd.DataFrame(index=METRIC_INDEX_ORDER)  # 性能评估结果汇总

    for alg_fn in list(zip(algorithm_list, process_fn_list)):
        alg = alg_fn[0]
        process_fn = alg_fn[1]

        print(f'------ {alg} ------')
        print(f'------ {alg} ------', file=doc_out_file)

        rst_info = process_fn(data_dict=data_dict, doc_output=doc_out_file)

        result_dict[alg] = rst_info
        rst_df[alg] = rst_info.get('metric')

    rst_df.to_excel(excel_out_path, index=True)

    # 绘制 ROC 曲线
    plt.figure(figsize=(8, 6))
    for alg in algorithm_list:
        rst_info = result_dict.get(alg)
        fpr = rst_info.get('fpr')
        tpr = rst_info.get('tpr')
        roc_auc = rst_info.get('roc_auc')
        plt.plot(fpr, tpr, lw=2, label=f'{alg} (AUC={roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC for {dataset_name}')
    plt.legend(loc="lower right")
    fig_path = f'../output/{dataset_name}-roc-curve.png'
    if min_max_flag:
        fig_path = f'../output/{dataset_name}-MinMax-roc-curve.png'
    plt.savefig(fig_path)
    plt.close()

    # 绘制 PR 曲线
    plt.figure(figsize=(8, 6))
    for alg in algorithm_list:
        rst_info = result_dict.get(alg)
        precision = rst_info.get('precision')
        recall = rst_info.get('recall')
        avg_prec = rst_info.get('average-precision')
        plt.plot(recall, precision, lw=2, label=f'{alg} (AP={avg_prec:.3f})')
    # plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'PR Curve for {dataset_name}')
    plt.legend(loc="lower left")
    fig_path = f'../output/{dataset_name}-pr-curve.png'
    if min_max_flag:
        fig_path = f'../output/{dataset_name}-MinMax-pr-curve.png'
    plt.savefig(fig_path)
    plt.close()


def process_one_kan(data_dict, doc_output):
    X_train = data_dict.get('X_train')
    X_test = data_dict.get('X_test')
    y_train = data_dict.get('y_train')
    y_test = data_dict.get('y_test')

    # Convert DataFrame to PyTorch Tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    # Create TensorDatasets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor.view(-1, 1))
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor.view(-1, 1))

    # Create DataLoaders
    train_dataloader = DataLoader(train_dataset, batch_size=6, shuffle=True, num_workers=19, persistent_workers=True)
    test_dataloader = DataLoader(test_dataset, batch_size=6, shuffle=False, num_workers=19, persistent_workers=True)

    model = KAN(in_dim=X_train.shape[1])

    trainer = L.Trainer(max_epochs=100)
    trainer.fit(model, train_dataloader)
    # trainer.test(model, test_dataloader)

    model.eval()

    y_hat = model(X_test_tensor)

    y_proba = torch.sigmoid(y_hat)
    y_pred = y_proba.round().int()

    y_proba = y_proba.detach().numpy()
    y_pred = y_pred[:, 0].detach().numpy()

    rst_info = evaluate(
        y_true=y_test,
        y_pred=y_pred,
        y_score=y_proba,
        doc_output=doc_output
    )
    return rst_info


def process_one_mlp(data_dict, doc_output):
    X_train = data_dict.get('X_train')
    X_test = data_dict.get('X_test')
    y_train = data_dict.get('y_train')
    y_test = data_dict.get('y_test')

    # Convert DataFrame to PyTorch Tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    # Create TensorDatasets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor.view(-1, 1))
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor.view(-1, 1))

    # Create DataLoaders
    train_dataloader = DataLoader(train_dataset, batch_size=6, shuffle=True, num_workers=19, persistent_workers=True)
    test_dataloader = DataLoader(test_dataset, batch_size=6, shuffle=False, num_workers=19, persistent_workers=True)

    model = MLP(in_dim=X_train.shape[1])
    trainer = L.Trainer(max_epochs=100)
    trainer.fit(model, train_dataloader)
    # trainer.test(model, test_dataloader)

    model.eval()

    y_hat = model(X_test_tensor)

    y_proba = torch.sigmoid(y_hat)
    y_pred = y_proba.round().int()

    y_proba = y_proba.detach().numpy()
    y_pred = y_pred[:, 0].detach().numpy()

    rst_info = evaluate(
        y_true=y_test,
        y_pred=y_pred,
        y_score=y_proba,
        doc_output=doc_output
    )
    return rst_info


def process_one_xgb(data_dict, doc_output):
    X_train = data_dict.get('X_train')
    X_test = data_dict.get('X_test')
    y_train = data_dict.get('y_train')
    y_test = data_dict.get('y_test')

    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    rst_info = evaluate(
        y_true=y_test,
        y_pred=y_pred,
        y_score=y_proba,
        doc_output=doc_output
    )
    return rst_info


# RandomForest
def process_one_rf(data_dict, doc_output):
    X_train = data_dict.get('X_train')
    X_test = data_dict.get('X_test')
    y_train = data_dict.get('y_train')
    y_test = data_dict.get('y_test')

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    rst_info = evaluate(
        y_true=y_test,
        y_pred=y_pred,
        y_score=y_proba,
        doc_output=doc_output
    )
    return rst_info


# LogisticRegression
def process_one_lr(data_dict, doc_output):
    X_train = data_dict.get('X_train')
    X_test = data_dict.get('X_test')
    y_train = data_dict.get('y_train')
    y_test = data_dict.get('y_test')

    model = LogisticRegressionCV(cv=5, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    rst_info = evaluate(
        y_true=y_test,
        y_pred=y_pred,
        y_score=y_proba,
        doc_output=doc_output
    )
    return rst_info




