import warnings

from STransfer.params import ParamConfig
from STransfer.utils import *
from STransfer.modules import *
from STransfer.train import *
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.lines as mlines
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib

# environment
warnings.filterwarnings('ignore')
random_seed = 2023
fix_seed(random_seed)
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'  # gpu

root_path = r"C:\Users\YUXIN\Desktop\STransfer\STransfer\params\mPFC\BZ5-BZ14"
pair_list = [
    # ("BZ5", "BZ9"),
    ("BZ5", "BZ14"),
    # ("BZ9", "BZ14")
]

for slicename_s, slicename_t in pair_list:
    print(f"processing ：source={slicename_s} | target={slicename_t}")
    params = ParamConfig(slicename_s, slicename_t)
    n_top_genes = 140
    adata_s, labels_s, graph_dict_s = process_slice_starmap(slicename_s, root_path, device)
    adata_t, labels_t, graph_dict_t = process_slice_starmap(slicename_t, root_path, device)
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

    # matplotlib.rcParams['font.family'] = 'Times New Roman'
    # matplotlib.rcParams['font.size'] = 12
    #
    #
    # X_raw = np.concatenate([adata_s.obsm['process'], adata_t.obsm['process']], axis=0)
    # X_feat = np.concatenate([laten_z_s.detach().cpu().numpy(), laten_z_t.detach().cpu().numpy()], axis=0)
    # domain_labels = np.array([0] * len(labels_s) + [1] * len(labels_t))
    #
    #
    # tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    # X_raw_tsne = tsne.fit_transform(X_raw)
    #
    # tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    # X_feat_tsne = tsne.fit_transform(X_feat)
    #
    #
    # plt.figure(figsize=(14, 6))
    #
    # plt.subplot(1, 2, 1)
    # plt.scatter(X_raw_tsne[domain_labels == 0, 0], X_raw_tsne[domain_labels == 0, 1], label='Source', alpha=0.6)
    # plt.scatter(X_raw_tsne[domain_labels == 1, 0], X_raw_tsne[domain_labels == 1, 1], label='Target', alpha=0.6)
    # plt.title(" t-SNE of raw data")
    # plt.xlabel("Dim 1")
    # plt.ylabel("Dim 2")
    # # plt.legend()
    #
    #
    # plt.subplot(1, 2, 2)
    # plt.scatter(X_feat_tsne[domain_labels == 0, 0], X_feat_tsne[domain_labels == 0, 1], label='Source', alpha=0.6)
    # plt.scatter(X_feat_tsne[domain_labels == 1, 0], X_feat_tsne[domain_labels == 1, 1], label='Target', alpha=0.6)
    # plt.title("t-SNE of embedded feature")
    # plt.xlabel("Dim 1")
    # plt.ylabel("Dim 2")
    # # plt.legend()
    #
    # plt.tight_layout()
    # tsne_plot = os.path.join(params.model_root, params.source_clust_plot + "_tsne.png")
    # plt.savefig(tsne_plot, dpi=300)
    # plt.show()

    color_mapping = {
        0: "#5698D3",
        1: "#C2A16C",
        2: "#6F6DAF",
        3: "#8B2A2A",

    }
    legend_handles = [
        mlines.Line2D([], [], color=color, marker='o', linestyle='None', markersize=10, label=str(label))
        for label, color in color_mapping.items()
    ]
    adata_s.obs["x_pixel"] = adata_s.obsm["spatial"][:, 0]
    adata_s.obs["y_pixel"] = adata_s.obsm["spatial"][:, 1]
    # true
    adata_s.obs['true'] = labels_s.cpu()
    adata_s.obs["color_true"] = adata_s.obs['true'].map(color_mapping)
    fig, axes = plt.subplots(1, 2, figsize=(20, 15))
    ax1 = axes[0]

    scatter1 = ax1.scatter(adata_s.obs["y_pixel"], adata_s.obs["x_pixel"],
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

    # pred
    true_label1 = pred_labels_s
    adata_s.obs['pred'] = true_label1
    adata_s.obs["color"] = adata_s.obs['pred'].map(color_mapping)

    ax2 = axes[1]  # 3列中的第二列
    scatter2 = ax2.scatter(adata_s.obs["y_pixel"], adata_s.obs["x_pixel"],
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

    ari_text = f"ARI: {ARI_s:.3f}"
    acc_text = f"ACC: {ACC_s:.3f}"
    ax2.text(0.25, 1.02, ari_text, ha='left', va='center', transform=ax2.transAxes, fontsize=20, color='black')
    ax2.text(0.55, 1.02, acc_text, ha='left', va='center', transform=ax2.transAxes, fontsize=20, color='black')


    plt.tight_layout()

    source_clust_plot = os.path.join(params.model_root, params.source_clust_plot + ".png")
    plt.savefig(source_clust_plot, dpi=300)
    plt.close()
    plt.show()


    fig, ax = plt.subplots(figsize=(10, 10))
    scatter = ax.scatter(adata_s.obs["y_pixel"], adata_s.obs["x_pixel"],
                         c=adata_s.obs["color_true"],
                         s=(900000 / adata_s.shape[0]) * 0.25,
                         alpha=0.8, edgecolor='black', linewidths=0.09)

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

    adata_s.obs['pred'] = pred_labels_s
    adata_s.obs["color"] = adata_s.obs['pred'].map(color_mapping)

    fig, ax = plt.subplots(figsize=(10, 10))
    scatter = ax.scatter(adata_s.obs["y_pixel"], adata_s.obs["x_pixel"],
                         c=adata_s.obs["color"],
                         s=(900000 / adata_s.shape[0]) * 0.25,
                         alpha=0.8, edgecolor='black', linewidths=0.09)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for spine in ['top', 'right', 'left', 'bottom']:
        ax.spines[spine].set_linewidth(0.8)
        ax.spines[spine].set_color("black")
    ax.set_aspect('equal', 'box')
    ax.grid(False)

    ari_text = f"ARI: {ARI_s:.3f}"
    acc_text = f"ACC: {ACC_s:.3f}"
    ax.text(0.08, 1.02, ari_text, ha='left', va='center', transform=ax.transAxes, fontsize=20, color='black')
    ax.text(0.58, 1.02, acc_text, ha='left', va='center', transform=ax.transAxes, fontsize=20, color='black')

    plt.tight_layout()

    source_clust_plot_pred = os.path.join(params.model_root, params.source_clust_plot + "_pred.png")
    plt.savefig(source_clust_plot_pred, dpi=300)
    plt.close()
    plt.show()


    ###############################target###
    adata_t.obs['x_pixel'] = adata_t.obsm['spatial'][:, 0]
    adata_t.obs['y_pixel'] = adata_t.obsm['spatial'][:, 1]

    # true
    adata_t.obs['true'] = labels_t.cpu()
    adata_t.obs["color_true"] = adata_t.obs['true'].map(color_mapping)
    fig, axes = plt.subplots(1, 2, figsize=(20, 15))
    ax1 = axes[0]

    scatter1 = ax1.scatter(adata_t.obs["y_pixel"], adata_t.obs["x_pixel"], c=adata_t.obs["color_true"],
                           s=(900000 / adata_t.shape[0]) * 0.25, alpha=0.8, edgecolor='black', linewidths=0.09)

    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)


    for spine in ['top', 'right', 'left', 'bottom']:
        ax1.spines[spine].set_linewidth(0.8)
        ax1.spines[spine].set_color("black")

    ax1.set_aspect('equal', 'box')
    ax1.grid(False)

    # pred
    true_label1 = pred_labels_t
    adata_t.obs['pred'] = true_label1
    adata_t.obs["color"] = adata_t.obs['pred'].map(color_mapping)

    ax2 = axes[1]
    scatter2 = ax2.scatter(adata_t.obs["y_pixel"], adata_t.obs["x_pixel"], c=adata_t.obs["color"],
                           s=(900000 / adata_t.shape[0]) * 0.25, alpha=0.8, edgecolor='black', linewidths=0.09)

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
    plt.close()
    plt.show()



    adata_t.obs['x_pixel'] = adata_t.obsm['spatial'][:, 0]
    adata_t.obs['y_pixel'] = adata_t.obsm['spatial'][:, 1]
    adata_t.obs['true'] = labels_t.cpu()
    adata_t.obs["color_true"] = adata_t.obs['true'].map(color_mapping)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(adata_t.obs["y_pixel"], adata_t.obs["x_pixel"],
               c=adata_t.obs["color_true"],
               s=(900000 / adata_t.shape[0]) * 0.25,
               alpha=0.8, edgecolor='black', linewidths=0.09)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    for spine in ['top', 'right', 'left', 'bottom']:
        ax.spines[spine].set_linewidth(0.8)
        ax.spines[spine].set_color("black")

    ax.set_aspect('equal', 'box')
    ax.grid(False)


    plt.tight_layout()
    target_clust_plot_true = os.path.join(params.model_root, params.target_clust_plot + "_true.png")
    plt.savefig(target_clust_plot_true, dpi=300)
    plt.close()
    # plt.show()

    adata_t.obs['pred'] = pred_labels_t
    adata_t.obs["color"] = adata_t.obs['pred'].map(color_mapping)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(adata_t.obs["y_pixel"], adata_t.obs["x_pixel"],
               c=adata_t.obs["color"],
               s=(900000 / adata_t.shape[0]) * 0.25,
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
    ax.text(0.08, 1.02, ari_text, ha='left', va='center', transform=ax.transAxes, fontsize=20, color='black')
    ax.text(0.58, 1.02, acc_text, ha='left', va='center', transform=ax.transAxes, fontsize=20, color='black')
    # ax.invert_yaxis()

    plt.tight_layout()
    target_clust_plot_pred = os.path.join(params.model_root, params.target_clust_plot + "_pred.png")
    plt.savefig(target_clust_plot_pred, dpi=300)
    plt.close()
    plt.show()