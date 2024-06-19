import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import auc
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score


METRIC_INDEX_ORDER = [
    'TP', 'TN', 'FP', 'FN', 'test-size', 'test-pos', 'test-neg',
    'accuracy', 'recall-pos', 'recall-neg', 'F1-pos', 'F1-neg',
    'precision-pos', 'precision-neg', 'average-precision', 'PR-AUC', 'ROC-AUC'
]


def evaluate(y_true, y_pred, y_score, doc_output):
    accuracy = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print(f'Accuracy: {accuracy:.4f}', file=doc_output)
    print(f'Recall: {recall:.4f}', file=doc_output)
    print(f'Precision: {precision:.4f}', file=doc_output)
    print(f'F1-score: {f1:.4f}', file=doc_output)

    prfs = precision_recall_fscore_support(y_true, y_pred)
    print('precision_recall_fscore_support:', file=doc_output)
    print(prfs, file=doc_output)

    report = classification_report(y_true, y_pred, digits=4)
    print('classification report:', file=doc_output)
    print(report, file=doc_output)

    # 绘制混淆矩阵图
    cm = confusion_matrix(y_true, y_pred)

    print('confusion matrix:', file=doc_output)
    print(cm, file=doc_output)

    # 绘制 ROC 曲线
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    print(f'ROC AUC: {roc_auc:.4f}', file=doc_output)

    # 绘制 PR 曲线
    precision_values, recall_values, _ = precision_recall_curve(y_true, y_score)
    avg_precision = average_precision_score(y_true, y_score)
    pr_auc = np.trapz(precision_values[::-1], recall_values[::-1])  # PR 曲线下方面积

    print(f'Average Precision: {avg_precision:.4f}', file=doc_output)

    rst_series = pd.Series(dtype=str, index=METRIC_INDEX_ORDER)
    rst_series['TP'] = str(cm[1, 1])
    rst_series['TN'] = str(cm[0, 0])
    rst_series['FP'] = str(cm[0, 1])
    rst_series['FN'] = str(cm[1, 0])
    rst_series['test-size'] = str(np.sum(cm))
    rst_series['test-pos'] = str(cm[1, 0] + cm[1, 1])
    rst_series['test-neg'] = str(cm[0, 0] + cm[0, 1])
    rst_series['accuracy'] = f'{accuracy:.4f}'
    rst_series['precision-neg'] = f'{prfs[0][0]:.4f}'
    rst_series['precision-pos'] = f'{prfs[0][1]:.4f}'
    rst_series['recall-neg'] = f'{prfs[1][0]:.4f}'
    rst_series['recall-pos'] = f'{prfs[1][1]:.4f}'
    rst_series['F1-neg'] = f'{prfs[2][0]:.4f}'
    rst_series['F1-pos'] = f'{prfs[2][1]:.4f}'
    rst_series['average-precision'] = f'{avg_precision:.4f}'
    rst_series['ROC-AUC'] = f'{roc_auc:.4f}'
    rst_series['PR-AUC'] = f'{pr_auc:.4f}'

    rst_info = {
        'fpr': fpr,
        'tpr': tpr,
        'roc_auc': roc_auc,
        'precision': precision_values,
        'recall': recall_values,
        'average-precision': avg_precision,
        'metric': rst_series
    }

    # 返回 ROC 曲线和 PR 曲线画图所需数据，用于多个算法画图比较
    # 返回性能评估结果 rst_series
    return rst_info


def evaluate_plot(y_true, y_pred, y_score, doc_output, fig_folder_path, dataset_name, algorithm):
    accuracy = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print(f'Accuracy: {accuracy:.4f}', file=doc_output)
    print(f'Recall: {recall:.4f}', file=doc_output)
    print(f'Precision: {precision:.4f}', file=doc_output)
    print(f'F1-score: {f1:.4f}', file=doc_output)

    prfs = precision_recall_fscore_support(y_true, y_pred)
    print('precision_recall_fscore_support:', file=doc_output)
    print(prfs, file=doc_output)

    report = classification_report(y_true, y_pred, digits=4)
    print('classification report:', file=doc_output)
    print(report, file=doc_output)

    # 绘制混淆矩阵图
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix for {dataset_name} with {algorithm}')
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Negative', 'Positive'])
    plt.yticks(tick_marks, ['Negative', 'Positive'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    fig_path = os.path.join(fig_folder_path, f'{dataset_name}_{algorithm}_confusion_matrix.png')
    plt.savefig(fig_path)
    plt.close()

    print('confusion matrix:', file=doc_output)
    print(cm, file=doc_output)

    # 绘制 ROC 曲线
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC for {dataset_name} with {algorithm}')
    plt.legend(loc="lower right")
    fig_path = os.path.join(fig_folder_path, f'{dataset_name}_{algorithm}_roc_curve.png')
    plt.savefig(fig_path)
    plt.close()

    print(f'ROC AUC: {roc_auc:.4f}', file=doc_output)

    # 绘制 PR 曲线
    precision_values, recall_values, _ = precision_recall_curve(y_true, y_score)
    avg_precision = average_precision_score(y_true, y_score)
    pr_auc = np.trapz(precision_values[::-1], recall_values[::-1])  # PR 曲线下方面积
    plt.figure(figsize=(8, 6))
    plt.plot(recall_values, precision_values, color='b', lw=2, label=f'Average Precision = {avg_precision:.2f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title(f'PR Curve for {dataset_name} with {algorithm}')
    plt.legend(loc="lower left")
    fig_path = os.path.join(fig_folder_path, f'{dataset_name}_{algorithm}_pr_curve.png')
    plt.savefig(fig_path)
    plt.close()

    print(f'Average Precision: {avg_precision:.4f}', file=doc_output)

    rst_series = pd.Series(dtype=str, index=METRIC_INDEX_ORDER)
    rst_series['TP'] = str(cm[1, 1])
    rst_series['TN'] = str(cm[0, 0])
    rst_series['FP'] = str(cm[0, 1])
    rst_series['FN'] = str(cm[1, 0])
    rst_series['test-size'] = str(np.sum(cm))
    rst_series['test-pos'] = str(cm[1, 0] + cm[1, 1])
    rst_series['test-neg'] = str(cm[0, 0] + cm[0, 1])
    rst_series['accuracy'] = f'{accuracy:.4f}'
    rst_series['precision-neg'] = f'{prfs[0][0]:.4f}'
    rst_series['precision-pos'] = f'{prfs[0][1]:.4f}'
    rst_series['recall-neg'] = f'{prfs[1][0]:.4f}'
    rst_series['recall-pos'] = f'{prfs[1][1]:.4f}'
    rst_series['F1-neg'] = f'{prfs[2][0]:.4f}'
    rst_series['F1-pos'] = f'{prfs[2][1]:.4f}'
    rst_series['average-precision'] = f'{avg_precision:.4f}'
    rst_series['ROC-AUC'] = f'{roc_auc:.4f}'
    rst_series['PR-AUC'] = f'{pr_auc:.4f}'

    rst_info = {
        'fpr': fpr,
        'tpr': tpr,
        'roc_auc': roc_auc,
        'precision': precision_values,
        'recall': recall_values,
        'average-precision': avg_precision,
        'metric': rst_series
    }

    # 返回 ROC 曲线和 PR 曲线画图所需数据，用于多个算法画图比较
    # 返回性能评估结果 rst_series
    return rst_info


