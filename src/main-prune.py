import warnings
import os

import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import torch

from kan import *

import MedicalDataLoader as mdl


def kan_prune(name, X_train, X_test, y_train, y_test):
    # Convert DataFrame to PyTorch Tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    formula_path = f'../output/formula/{name}/'
    if not os.path.exists(formula_path):
        os.makedirs(formula_path)

    pruned_net_path = f'../output/pruned_figs/'
    if not os.path.exists(pruned_net_path):
        os.makedirs(pruned_net_path)

    model = MultKAN(width=[X_train.shape[1], 1])

    # Draw the initial network diagram
    plt.figure()
    model(X_train_tensor)
    model.plot(beta=100, folder=formula_path)
    fig_path = f'../output/{name}_initial_plot.png'
    plt.savefig(fig_path)
    plt.close()

    dataset = {
        'train_input': X_train_tensor,
        'train_label': y_train_tensor,
        'test_input': X_test_tensor,
        'test_label': y_test_tensor
    }

    def train_acc():
        return torch.mean((torch.round(model(X_train_tensor)) == y_train_tensor).float())

    def test_acc():
        return torch.mean((torch.round(model(X_test_tensor)) == y_test_tensor).float())

    _ = model.fit(dataset, opt="LBFGS", metrics=(train_acc, test_acc), steps=20, lamb=0.01, lamb_entropy=10.)

    # Draw the trained network diagram
    plt.figure()
    model(X_train_tensor)
    model.plot(folder=formula_path, scale=1.0)
    eps_fig_path = f'../output/{name}_trained_plot.eps'
    plt.savefig(eps_fig_path, format='eps', dpi=400)
    png_fig_path = f'../output/{name}_trained_plot.png'
    plt.savefig(png_fig_path, format='png', dpi=400)
    plt.close()

    # Plotting the pruned network
    plt.figure()
    model = model.prune(node_th=0.01, edge_th=0.01)
    model(X_train_tensor)
    model.plot(folder=formula_path, scale=1.0)
    fig_path = os.path.join(pruned_net_path, f'{name}_prune_plot.eps')
    plt.savefig(fig_path, format='eps', dpi=400)
    png_fig_path = os.path.join(pruned_net_path, f'{name}_prune_plot.png')
    plt.savefig(png_fig_path, format='png', dpi=400)
    pdf_fig_path = os.path.join(pruned_net_path, f'{name}_prune_plot.pdf')
    plt.savefig(pdf_fig_path, format='pdf', dpi=400)
    plt.close()

    lib = ['x', 'x^2', 'x^3', 'x^4', 'exp', 'log', 'sqrt', 'tanh', 'sin', 'tan', 'abs']
    model.auto_symbolic(lib=lib)

    formula1 = model.symbolic_formula()[0]

    formula_file_folder = '../output/formula_files'
    if not os.path.exists(formula_file_folder):
        os.makedirs(formula_file_folder)
    formula_file = open(f'../output/formula_files/formula_file_{name}.txt', 'w')
    print(formula1, file=formula_file)
    formula_file.close()

    formula_path = f'../output/formula-symbol/{name}/'
    if not os.path.exists(formula_path):
        os.makedirs(formula_path)

    symbol_fig_path = f'../output/symbol_figs/'
    if not os.path.exists(symbol_fig_path):
        os.makedirs(symbol_fig_path)

    # plt.figure()
    model.plot(folder=formula_path)
    fig_path = os.path.join(symbol_fig_path, f'{name}_symbol_plot.eps')
    plt.savefig(fig_path, format='eps', dpi=400)
    png_fig_path = os.path.join(symbol_fig_path, f'{name}_symbol_plot.png')
    plt.savefig(png_fig_path, format='png', dpi=400)
    pdf_fig_path = os.path.join(symbol_fig_path, f'{name}_symbol_plot.pdf')
    plt.savefig(pdf_fig_path, format='pdf', dpi=400)
    plt.close()


def prune_all(dataset_name_list, data_path_list, data_load_fn_list):
    for info in list(zip(dataset_name_list, data_path_list, data_load_fn_list)):
        dataset_name = info[0]
        data_path = info[1]
        data_load_fn = info[2]

        # if 'risk' not in dataset_name:
        #     continue

        print(f'------ {dataset_name} ------')

        X, y = data_load_fn(file_path=data_path)  # ndarray

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            stratify=y,
            random_state=42
        )

        # 数据 Min-Max 预处理
        flag = False
        if dataset_name in ['breast-cancer', 'diabetes_2', 'sepsis']:
            flag = True
        if flag:
            scaler = MinMaxScaler(feature_range=(0, 1))
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        kan_prune(dataset_name, X_train, X_test, y_train, y_test)


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

    prune_all(dataset_name_list=names, data_path_list=path_list, data_load_fn_list=data_load_fn_list)



