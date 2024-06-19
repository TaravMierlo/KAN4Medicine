import os.path

import numpy as np
import matplotlib.pyplot as plt


def plot_network_structure(folder_path, name, in_dim, mask):
    scale = 1.0
    out_dim = 1
    width = [in_dim, out_dim]  # [in_dim, 1]
    neuron_depth = len(width)  # 2

    min_spacing = 1.0 / np.maximum(np.max(width), 5)

    y0 = 0.4
    y1 = y0 / np.maximum(in_dim, 3)

    fig, ax = plt.subplots(figsize=(10 * scale, 10 * scale * y0))

    # step 1: plot scatters and lines
    for l in range(neuron_depth):  # l (layer): [0, 1]
        n = width[l]  # 第 l 层的神经元个数

        for i in range(n):
            plt.scatter(1 / (2 * n) + i / n, l * y0, s=min_spacing ** 2 * 10000 * scale ** 2, color='black')

            if l < neuron_depth - 1:  # l 不是最后一层
                n_next = width[l + 1]  # out_dim = 1，下一层的神经元个数
                N = n * n_next  # = n

                for j in range(n_next):  # j: [0]
                    id_ = i * n_next + j
                    if mask[id_] == 0:
                        continue

                    # plot connection lines between input nodes and formula figs
                    plt.plot(
                        [1 / (2 * n) + i / n, 1 / (2 * N) + id_ / N],
                        [l * y0, (l + 1 / 2) * y0 - y1],
                        lw=2 * scale,
                        color='black'
                    )

                    # plot connection lines between next layer (output) nodes and formula figs
                    plt.plot(
                        [1 / (2 * N) + id_ / N, 1 / (2 * n_next) + j / n_next],
                        [(l + 1 / 2) * y0 + y1, (l + 1) * y0],
                        lw=2 * scale,
                        color='black'
                    )

        plt.xlim(0, 1)
        plt.ylim(-0.1 * y0, (neuron_depth - 1 + 0.1) * y0)

    # step 2: plot splines

    plt.axis('off')

    # -- Transformation functions
    DC_to_FC = ax.transData.transform
    FC_to_NFC = fig.transFigure.inverted().transform
    # -- Take data coordinates and transform them to normalized figure coordinates
    DC_to_NFC = lambda x: FC_to_NFC(DC_to_FC(x))

    l = 0
    n = in_dim

    # step 2: plot splines
    for i in range(n):  # [0, 1, ..., in_dim-1]
        n_next = out_dim  # = 1
        N = n * n_next  # = n
        for j in range(n_next):  # j: [0]
            id_ = i * n_next + j
            if mask[id_] == 0:
                continue

            im = plt.imread(f'{folder_path}/sp_{l}_{i}_{j}.png')

            left = DC_to_NFC([1 / (2 * N) + id_ / N - y1, 0])[0]
            right = DC_to_NFC([1 / (2 * N) + id_ / N + y1, 0])[0]
            bottom = DC_to_NFC([0, (l + 1 / 2) * y0 - y1])[1]
            up = DC_to_NFC([0, (l + 1 / 2) * y0 + y1])[1]
            newax = fig.add_axes([left, bottom, right - left, up - bottom])
            # newax.imshow(im, alpha=alpha[l][j][i])
            newax.imshow(im)
            newax.axis('off')

    manual_plot_folder = f'../output/manual-plot/'
    if not os.path.exists(manual_plot_folder):
        os.makedirs(manual_plot_folder)

    png_path = f'../output/manual-plot/{name}_prune_plot.png'
    eps_path = f'../output/manual-plot/{name}_prune_plot.eps'
    pdf_path = f'../output/manual-plot/{name}_prune_plot.pdf'
    plt.savefig(png_path, format='png', dpi=400)
    plt.savefig(eps_path, format='eps', dpi=400)
    plt.savefig(pdf_path, format='pdf', dpi=400)

    # plt.show()
    plt.close()


if __name__ == '__main__':
    fig_folder_path = f'../output/formula/'

    dataset_name_list = [
        'breast-cancer',
        'cervical-cancer',
        'diabetes_risk',
        'sepsis'
    ]

    in_dim_list = [30, 33, 31, 8]

    dataset_mask_dict = {
        'breast-cancer': [0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1],
        'cervical-cancer': [0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1],
        'diabetes_risk': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        'sepsis': [1, 1, 0, 1, 0, 0, 1, 1]
    }

    datasets_cnt = len(dataset_name_list)
    for m in range(datasets_cnt):
        dataset_name = dataset_name_list[m]
        in_dim = in_dim_list[m]
        mask = dataset_mask_dict.get(dataset_name)
        ffp = os.path.join(fig_folder_path, dataset_name)

        plot_network_structure(folder_path=ffp, name=dataset_name, in_dim=in_dim, mask=mask)
