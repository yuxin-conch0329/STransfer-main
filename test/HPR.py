import warnings
import matplotlib.lines as mlines
from STransfer.params import ParamConfig
from STransfer.utils import *
from STransfer.modules import *
from STransfer.train import *
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib

# environment
warnings.filterwarnings('ignore')
random_seed = 2023
fix_seed(random_seed)
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'  # gpu
label_mapping = {'MPA': 1, 'MPN': 2, 'BST': 3, 'fx': 4, "PVH": 5, "PVT": 6, "V3": 7, 'PV': 0}
data_dir = r"C:\Users\YUXIN\Desktop\STransfer\STransfer\params\HPR\STransfer"
# all_slicename = ['25','26','27','28','29']
# all_targets = ['25','26','27','28','29']

all_slicename = ['28']
all_targets = ['28','29']


for slicename_s in all_slicename:
    for slicename_t in all_targets:
        if str(slicename_s) == str(slicename_t):
            continue
        print(f"processing：source={slicename_s} | target={slicename_t}")
    n_top_genes = 150
    params = ParamConfig(slicename_s, slicename_t)
    path = os.path.join(data_dir, f"{slicename_s}.h5ad")
    path_t = os.path.join(data_dir, f"{slicename_t}.h5ad")

    adata_s, labels_s, graph_dict_s = preprocess_slice_merfish(path, label_mapping=label_mapping, device=device)
    adata_t, labels_t, graph_dict_t = preprocess_slice_merfish(path_t, label_mapping=label_mapping, device=device)

    # Initialize source domain encoder and classifier
    src_encoder = STransferEncoder(input_dim=n_top_genes).to(device)
    src_classifier = STransferClassifier(input_dim=64).to(device)

    # Initialize target domain encoder and domain discriminator (critic)
    tgt_encoder = STransferEncoder(input_dim=n_top_genes).to(device)
    critic = Discriminator().to(device)

    # === Train source domain encoder and classifier ===
    print("=== Training classifier for source domain ===")
    print(">>> Source Encoder <<<")
    print(src_encoder)
    print(">>> Source Classifier <<<")

    # If pretrained models exist, load them
    if os.path.exists(params.src_encoder_path) and os.path.exists(params.src_classifier_path):
        print("Model parameter files detected, loading...")
        src_encoder.load_state_dict(torch.load(params.src_encoder_path), strict=False)
        src_classifier.load_state_dict(torch.load(params.src_classifier_path), strict=False)
    else:
        # If not, train the source model from scratch
        print("Model parameters not found, starting training...")
        src_encoder, src_classifier = train_src(src_encoder, src_classifier, adata_s, graph_dict_s, labels_s, params)

    # Evaluate the source encoder and classifier on the source domain
    print("=== Evaluating classifier for source domain ===")
    eval_src(src_encoder, src_classifier, adata_s, graph_dict_s, labels_s)

    # === Train target encoder via adversarial training (domain adaptation) === GAN
    print("=== Training encoder for target domain ===")
    print(">>> Target Encoder <<<")
    print(tgt_encoder)
    print(">>> Critic <<<")

    # If pretrained target encoder and critic exist, load them
    if os.path.exists(params.critic_path) and os.path.exists(params.tgt_encoder_path):
        print("Model parameter files detected, loading...")
        critic.load_state_dict(torch.load(params.critic_path), strict=False)
        tgt_encoder.load_state_dict(torch.load(params.tgt_encoder_path), strict=False)
    else:
        print("Model parameters not found, starting training...")
        tgt_encoder.load_state_dict(src_encoder.state_dict())
        tgt_encoder = train_tgt(src_encoder, tgt_encoder, critic, adata_s, graph_dict_s, adata_t, graph_dict_t, params)



    print(">>> source only <<<")
    ACC_s,ARI_s,pred_labels_s,laten_z_s = eval_tgt(src_encoder, src_classifier, adata_s,graph_dict_s,labels_s)
    pred_save_path_s = os.path.join(params.model_root, params.source_pred_labels + ".csv")
    save_predictions_to_csv(pred_labels_s, pred_save_path_s)
    # === Evaluation on the target domains ===
    # Evaluate performance of domain-adapted target encoder
    print(">>> domain adaption <<<")
    ACC_t,ARI_t, pred_labels_t,laten_z_t = eval_tgt(tgt_encoder, src_classifier, adata_t, graph_dict_t, labels_t)
    pred_t_save_path = os.path.join(params.model_root, params.target_pred_labels + ".csv")
    save_predictions_to_csv(pred_labels_t, pred_t_save_path)

    matplotlib.rcParams['font.family'] = 'Times New Roman'
    matplotlib.rcParams['font.size'] = 12


    X_raw = np.concatenate([adata_s.obsm['process'], adata_t.obsm['process']], axis=0)
    X_feat = np.concatenate([laten_z_s.detach().cpu().numpy(), laten_z_t.detach().cpu().numpy()], axis=0)
    domain_labels = np.array([0] * len(labels_s) + [1] * len(labels_t))


    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    X_raw_tsne = tsne.fit_transform(X_raw)

    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    X_feat_tsne = tsne.fit_transform(X_feat)


    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.scatter(X_raw_tsne[domain_labels == 0, 0], X_raw_tsne[domain_labels == 0, 1], label='Source', alpha=0.6)
    plt.scatter(X_raw_tsne[domain_labels == 1, 0], X_raw_tsne[domain_labels == 1, 1], label='Target', alpha=0.6)
    plt.title(" t-SNE of raw data")
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    # plt.legend()


    plt.subplot(1, 2, 2)
    plt.scatter(X_feat_tsne[domain_labels == 0, 0], X_feat_tsne[domain_labels == 0, 1], label='Source', alpha=0.6)
    plt.scatter(X_feat_tsne[domain_labels == 1, 0], X_feat_tsne[domain_labels == 1, 1], label='Target', alpha=0.6)
    plt.title("t-SNE of embedded feature")
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    # plt.legend()

    plt.tight_layout()
    tsne_plot = os.path.join(params.model_root, params.source_clust_plot + "_tsne.png")
    plt.savefig(tsne_plot, dpi=300)
    plt.show()

    # Define a color mapping for each of the 8 classes (class 0–7)
    color_mapping = {
        0: "#5698D3",
        1: "#C2A16C",
        2: "#6F6DAF",
        3: "#8B2A2A",
        4: "#A65B8D",
        5: "#E6C44D",
        6: "#66B2A6",
        7: "#D17CA3",}

    legend_handles = [
        mlines.Line2D([], [], color=color_mapping[i], marker='o', linestyle='None', markersize=10, label=str(i))
        for i in range(8)]

    adata_s.obs['x_pixel'] = adata_s.obsm['spatial'][:, 0]
    adata_s.obs['y_pixel'] = adata_s.obsm['spatial'][:, 1]
    #  true
    adata_s.obs['true'] = labels_s.cpu()
    adata_s.obs["color_true"] = adata_s.obs['true'].map(color_mapping)
    fig, axes = plt.subplots(1, 2, figsize=(15, 10))

    # Plot ground truth labels
    ax1 = axes[0]
    scatter1 = ax1.scatter(adata_s.obs["x_pixel"], adata_s.obs["y_pixel"],
                           c=adata_s.obs["color_true"],
                           s=(500000 / adata_s.shape[0]) * 0.25,
                           alpha=0.8, edgecolor='black', linewidths=0.09)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)


    for spine in ['top', 'right', 'left', 'bottom']:
        ax1.spines[spine].set_linewidth(0.8)
        ax1.spines[spine].set_color("black")

    ax1.set_aspect('equal', 'box')
    ax1.grid(False)
    ax1.spines['bottom'].set_color("black")

    # Plot predicted labels
    true_label1 = pred_labels_s
    adata_s.obs['pred'] = true_label1
    adata_s.obs["color"] = adata_s.obs['pred'].map(color_mapping)


    ax2 = axes[1]
    scatter2 = ax2.scatter(adata_s.obs["x_pixel"], adata_s.obs["y_pixel"],
                           c=adata_s.obs["color"],
                           s=(500000 / adata_s.shape[0]) * 0.25,
                           alpha=0.8, edgecolor='black', linewidths=0.09)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)


    for spine in ['top', 'right', 'left', 'bottom']:
        ax2.spines[spine].set_linewidth(0.8)
        ax2.spines[spine].set_color("black")

    ax2.set_aspect('equal', 'box')
    ax2.grid(False)

    # ARI & ACC
    ari_text = f"ARI: {ARI_s:.3f}"
    acc_text = f"ACC: {ACC_s:.3f}"
    ax2.text(0.25, 1.02, ari_text, ha='left', va='center', transform=ax2.transAxes, fontsize=20, color='black')
    ax2.text(0.55, 1.02, acc_text, ha='left', va='center', transform=ax2.transAxes, fontsize=20, color='black')

    plt.tight_layout()

    source_clust_plot = os.path.join(params.model_root, params.source_clust_plot + ".png")
    plt.savefig(source_clust_plot, dpi=300)
    # plt.close()
    plt.show()

    # ------- Plot true labels alone -------
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(
        adata_s.obs["x_pixel"], adata_s.obs["y_pixel"],
        c=adata_s.obs["color_true"],
        s=(500000 / adata_s.shape[0]) * 0.25,
        alpha=0.8, edgecolor='black', linewidths=0.09
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    for spine in ['top', 'right', 'left', 'bottom']:
        ax.spines[spine].set_linewidth(0.8)
        ax.spines[spine].set_color("black")

    ax.set_aspect('equal', 'box')
    ax.grid(False)


    plt.tight_layout()
    source_clust_plot_true = os.path.join(params.model_root, params.source_clust_plot + "_true.png")
    plt.savefig(source_clust_plot_true, dpi=300)
    plt.close()
    plt.show()

    # Predicted
    adata_s.obs['pred'] = pred_labels_s
    adata_s.obs["color"] = adata_s.obs['pred'].map(color_mapping)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(
        adata_s.obs["x_pixel"], adata_s.obs["y_pixel"],
        c=adata_s.obs["color"],
        s=(500000 / adata_s.shape[0]) * 0.25,
        alpha=0.8, edgecolor='black', linewidths=0.09
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    for spine in ['top', 'right', 'left', 'bottom']:
        ax.spines[spine].set_linewidth(0.8)
        ax.spines[spine].set_color("black")

    ax.set_aspect('equal', 'box')
    ax.grid(False)

    # ARI & ACC
    ari_text = f"ARI: {ARI_s:.3f}"
    acc_text = f"ACC: {ACC_s:.3f}"
    ax.text(0.25, 1.02, ari_text, ha='left', va='center', transform=ax.transAxes, fontsize=20, color='black')
    ax.text(0.55, 1.02, acc_text, ha='left', va='center', transform=ax.transAxes, fontsize=20, color='black')

    plt.tight_layout()
    source_clust_plot_pred = os.path.join(params.model_root, params.source_clust_plot + "_pred.png")
    plt.savefig(source_clust_plot_pred, dpi=300)
    plt.close()
    plt.show()
##########target

    adata_t.obs['x_pixel'] = adata_t.obsm['spatial'][:, 0]
    adata_t.obs['y_pixel'] = adata_t.obsm['spatial'][:, 1]
    # 第一张图: true 标签
    adata_t.obs['true'] = labels_t.cpu()
    adata_t.obs["color_true"] = adata_t.obs['true'].map(color_mapping)
    fig, axes = plt.subplots(1, 2, figsize=(15, 10))
    # ---- Left plot: Ground Truth ----
    ax1 = axes[0]
    scatter1 = ax1.scatter(adata_t.obs["x_pixel"], adata_t.obs["y_pixel"],
                           c=adata_t.obs["color_true"],
                           s=(500000 / adata_t.shape[0]) * 0.25,
                           alpha=0.8, edgecolor='black', linewidths=0.09)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)


    for spine in ['top', 'right', 'left', 'bottom']:
        ax1.spines[spine].set_linewidth(0.8)
        ax1.spines[spine].set_color("black")

    ax1.set_aspect('equal', 'box')
    ax1.grid(False)
    ax1.spines['bottom'].set_color("black")

    # ---- Right plot: Prediction ----
    true_label1 = pred_labels_t
    adata_t.obs['pred'] = true_label1
    adata_t.obs["color"] = adata_t.obs['pred'].map(color_mapping)


    ax2 = axes[1]
    scatter2 = ax2.scatter(adata_t.obs["x_pixel"], adata_t.obs["y_pixel"], c=adata_t.obs["color"],
                           s=(500000 / adata_t.shape[0]) * 0.25, alpha=0.8, edgecolor='black', linewidths=0.09)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)


    for spine in ['top', 'right', 'left', 'bottom']:
        ax2.spines[spine].set_linewidth(0.8)
        ax2.spines[spine].set_color("black")

    ax2.set_aspect('equal', 'box')
    ax2.grid(False)

    # ARI & ACC
    ari_text = f"ARI: {ARI_t:.3f}"
    acc_text = f"ACC: {ACC_t:.3f}"
    ax2.text(0.25, 1.02, ari_text, ha='left', va='center', transform=ax2.transAxes, fontsize=20, color='black')
    ax2.text(0.55, 1.02, acc_text, ha='left', va='center', transform=ax2.transAxes, fontsize=20, color='black')

    plt.tight_layout()

    target_clust_plot = os.path.join(params.model_root, params.target_clust_plot + ".png")
    plt.savefig(target_clust_plot, dpi=300)
    # plt.close()
    plt.show()

    # ---------- SECOND FIGURE: Ground Truth Only ----------
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(adata_t.obs["x_pixel"], adata_t.obs["y_pixel"],
               c=adata_t.obs["color_true"],
               s=(500000 / adata_t.shape[0]) * 0.25,
               alpha=0.8, edgecolor='black', linewidths=0.09)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    for spine in ['top', 'right', 'left', 'bottom']:
        ax.spines[spine].set_linewidth(0.8)
        ax.spines[spine].set_color("black")

    ax.set_aspect('equal', 'box')
    ax.grid(False)

    # Save ground truth only plot
    true_clust_plot = os.path.join(params.model_root, params.target_clust_plot + "_true.png")
    plt.tight_layout()
    plt.savefig(true_clust_plot, dpi=300)
    plt.close()
    plt.show()

    # ---------- THIRD FIGURE: Prediction Only ----------
    adata_t.obs['pred'] = pred_labels_t
    adata_t.obs["color"] = adata_t.obs['pred'].map(color_mapping)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(adata_t.obs["x_pixel"], adata_t.obs["y_pixel"],
               c=adata_t.obs["color"],
               s=(500000 / adata_t.shape[0]) * 0.25,
               alpha=0.8, edgecolor='black', linewidths=0.09)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    for spine in ['top', 'right', 'left', 'bottom']:
        ax.spines[spine].set_linewidth(0.8)
        ax.spines[spine].set_color("black")

    ax.set_aspect('equal', 'box')
    ax.grid(False)


    ari_text = f"ARI: {ARI_t:.3f}"
    acc_text = f"ACC: {ACC_t:.3f}"
    ax.text(0.25, 1.02, ari_text, ha='left', va='center', transform=ax.transAxes, fontsize=20, color='black')
    ax.text(0.55, 1.02, acc_text, ha='left', va='center', transform=ax.transAxes, fontsize=20, color='black')

    # pred
    pred_clust_plot = os.path.join(params.model_root, params.target_clust_plot + "_pred.png")
    plt.tight_layout()
    plt.savefig(pred_clust_plot, dpi=300)
    plt.close()
    plt.show()
