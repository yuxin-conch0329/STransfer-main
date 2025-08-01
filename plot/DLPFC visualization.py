import matplotlib.lines as mlines
from params import ParamConfig
import matplotlib.pyplot as plt
import warnings
from sklearn.metrics import accuracy_score
from sklearn.metrics import adjusted_rand_score
from utils import *
warnings.filterwarnings("ignore")

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

pair_list = [("151507", "151508")]
data_dir =r"C:\Users\YUXIN\Desktop\new\new\data"
label_mapping = {'L1': 1, 'L2': 2, 'L3': 3, 'L4': 4, 'L5': 5, 'L6': 6, 'WM': 0}

for slicename_s, slicename_t in pair_list:
    params = ParamConfig(slicename_s, slicename_t)
    adata_s, data_s, labels_s, graph_dict_s = preprocess_slice_DLPFC(slicename_s, data_dir, device, label_mapping,1000)

    pred_save_path = os.path.join(params.model_root, params.source_pred_labels + ".csv")
    pred_labels_list = pd.read_csv(pred_save_path)
    pred_label = pred_labels_list['pred']

    ARI_s = adjusted_rand_score(pred_label, labels_s.cpu().numpy())
    acc_s = accuracy_score(pred_label, labels_s.cpu().numpy())
    print("ARI_s:", ARI_s)
    print("acc_s", acc_s)

    #visualization
    color_mapping = {
        0: "#5698D3",
        1: "#C2A16C",
        2: "#6F6DAF",
        3: "#8B2A2A",
        4: "#A65B8D",
        5: "#E6C44D",
        6: "#66B2A6",
    }
    legend_handles = [
        mlines.Line2D([], [], color=color, marker='o', linestyle='None', markersize=10, label=str(label))
        for label, color in color_mapping.items()]
    adata_s.obs['x_pixel'] = data_s['pxl_row_in_fullres'].values
    adata_s.obs['y_pixel'] = data_s['pxl_col_in_fullres'].values
    # true label
    adata_s.obs['true'] = labels_s.cpu()
    adata_s.obs["color_true"] = adata_s.obs['true'].map(color_mapping)
    fig, axes = plt.subplots(1, 2, figsize=(20, 15))
    ax1 = axes[0]
    scatter1 = ax1.scatter(adata_s.obs["y_pixel"], adata_s.obs["x_pixel"],
                           c=adata_s.obs["color_true"],
                           s=(900000 / adata_s.shape[0]) * 0.6,
                           alpha=0.8, edgecolor='black', linewidths=0.09)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for spine in ['top', 'right', 'left', 'bottom']:
        ax1.spines[spine].set_linewidth(0.8)
        ax1.spines[spine].set_color("black")

    ax1.set_aspect('equal', 'box')
    ax1.grid(False)

    # pred label
    true_label1 = pred_label.values
    adata_s.obs['pred'] = true_label1
    adata_s.obs["color"] = adata_s.obs['pred'].map(color_mapping)

    ax2 = axes[1]
    scatter2 = ax2.scatter(adata_s.obs["y_pixel"], adata_s.obs["x_pixel"],
                           c=adata_s.obs["color"],
                           s=(900000 / adata_s.shape[0]) * 0.6,
                           alpha=0.8, edgecolor='black', linewidths=0.09)

    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    for spine in ['top', 'right', 'left', 'bottom']:
        ax2.spines[spine].set_linewidth(0.8)
        ax2.spines[spine].set_color("black")

    ax2.set_aspect('equal', 'box')
    ax2.grid(False)

    ari_text = f"ARI: {ARI_s:.3f}"
    acc_text = f"ACC: {acc_s:.3f}"
    ax2.text(0.25, 1.02, ari_text, ha='left', va='center', transform=ax2.transAxes, fontsize=20, color='black')
    ax2.text(0.55, 1.02, acc_text, ha='left', va='center', transform=ax2.transAxes, fontsize=20, color='black')

    ax1.invert_yaxis()
    ax2.invert_yaxis()
    plt.tight_layout()
    plt.show()

