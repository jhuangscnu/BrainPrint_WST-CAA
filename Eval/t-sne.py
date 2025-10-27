import os
import numpy as np
import torch
import pickle
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib.cm as cm
from matplotlib import rcParams
from matplotlib import font_manager as fm
import matplotlib.gridspec as gridspec


font_files = [
    "../../.local/share/fonts/times.ttf",
    "../../.local/share/fonts/timesi.ttf",
    "../../.local/share/fonts/timesbd.ttf",
    "../../.local/share/fonts/timesbi.ttf"
]
for f in font_files:
    fm.fontManager.addfont(f)

# 全局字体设置为 Times New Roman
rcParams['font.family'] = 'Times New Roman'
rcParams['font.size'] = 16           # 默认字体大小，可调整
rcParams['axes.titlesize'] = 16      # 标题字体大小
rcParams['axes.labelsize'] = 14      # 坐标轴字体大小
rcParams['xtick.labelsize'] = 14     # x 轴刻度字体大小
rcParams['ytick.labelsize'] = 14     # y 轴刻度字体大小

# ==========================================================
# 🧩 通用的 t-SNE 可视化子图函数
# ==========================================================
def plot_tsne_subplot(ax, X, y, title='t-SNE', cmap_name='tab20', random_state=42):
    """
    在指定的 ax 上绘制 t-SNE 结果。
    - X: 输入数据
    - y: 标签（例如 Subject 编号）
    - title: 图标题（如 '(a) Raw Data t-SNE'）
    """
    # 标准化
    X = StandardScaler().fit_transform(X)
    tsne = TSNE(n_components=2, perplexity=30, init='pca', n_iter=1000, random_state=random_state)
    X_tsne = tsne.fit_transform(X)

    # 生成颜色映射
    num_classes = len(np.unique(y))
    cmap = cm.get_cmap(cmap_name, num_classes)

    # 绘制散点
    for lab in np.unique(y):
        idx = y == lab
        color = cmap(lab / num_classes)
        ax.scatter(X_tsne[idx, 0], X_tsne[idx, 1], s=8, color=color)

    # 调整刻度与线条样式
    ax.tick_params(axis='both', which='major', labelsize=18, width=1.5, length=6)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    # 去除坐标轴名称
    ax.set_xlabel('')
    ax.set_ylabel('')

    # 将标题放在下方
    ax.set_title(title, fontsize=24, y=-0.18)  # y 越小越靠下


# ==========================================================
# 📊 主程序：加载数据并绘图
# ==========================================================
def main():
    # ------------------------
    # 路径设置
    # ------------------------
    data_dir = '../AfterWST/DEAP/'              # WST 后的数据
    raw_data_dir = '../ProcessedData/DEAP_NoPre/'  # 原始数据

    # 文件列表
    pwst_data_files = [os.path.join(data_dir, f'sub{s:02d}WST.pt') for s in range(1, 33)]
    raw_data_files = [os.path.join(raw_data_dir, f's{s:02d}.pkl') for s in range(1, 33)]

    # ------------------------
    # 加载 WST 后数据
    # ------------------------
    print('Loading WST processed data...')
    pwst_all, labels_all = [], []

    for subj_idx, file_path in enumerate(pwst_data_files):
        pwst_tensor = torch.load(file_path)  # shape [40, 32, 175, 24]
        pwst_flat = pwst_tensor.cpu().numpy().reshape(pwst_tensor.shape[0], -1)
        pwst_all.append(pwst_flat)
        labels_all.append(np.full(pwst_tensor.shape[0], subj_idx + 1))

    X_pwst = np.vstack(pwst_all)
    y_pwst = np.hstack(labels_all)

    # ------------------------
    # 加载原始数据
    # ------------------------
    print('Loading Raw EEG data...')
    raw_all, raw_labels = [], []

    for subj_idx, file_path in enumerate(raw_data_files):
        with open(file_path, 'rb') as f:
            raw_data = pickle.load(f)  # shape [samples, channels, points]
        raw_flat = raw_data.reshape(raw_data.shape[0], -1)
        raw_all.append(raw_flat)
        raw_labels.append(np.full(raw_data.shape[0], subj_idx + 1))

    X_raw = np.vstack(raw_all)
    y_raw = np.hstack(raw_labels)

    # ------------------------
    # 绘制两张图 (Raw vs PWST)
    # ------------------------
    print('Running t-SNE visualization...')

    fig = plt.figure(figsize=(13.5, 6))
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.15)  # wspace 越小，子图越靠近

    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])

    plot_tsne_subplot(ax0, X_raw, y_raw, title='(a) Raw Data t-SNE')
    plot_tsne_subplot(ax1, X_pwst, y_pwst, title='(b) PWST t-SNE')

    plt.tight_layout(pad=2.0)
    plt.savefig('DEAP_tSNE_compare.png', dpi=600, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    main()