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
pair_list = [("CRC_s6_rep2", "CRC_s6_rep1"),]
label_mapping = {'tumor': 0,
    'tumor_stroma_ic_med_to_high': 0,
    'stroma_fibroblastic_ic_high': 1,
    'stroma_fibroblastic_ic_med': 1,
    'stroma_desmoplastic_ic_low': 1,
    'stroma_desmoplastic_ic_med_to_high': 1,
    'epithelium_submucosa': 2,
    'non_neo_epithelium': 2,
    'submucosa': 3,
    'IC_aggregate_submucosa': 3,
    'muscularis_ic_med_to_high': 4,
    'IC_aggregate_stroma_or_muscularis': 4,
    'exclude': 5}

base_path = r"C:\E\JSU\BIO\file\STransfer\STrafer\datanew\CRC"

for slicename_t, slicename_s in pair_list:
    print(f"processing：source={slicename_s} | target={slicename_t}")
    params = ParamConfig(slicename_s, slicename_t)
    adata_s, labels_s = preprocess_slice_CRC(base_path,slicename_s,label_mapping=label_mapping, device=device)
    adata_t, labels_t = preprocess_slice_CRC(base_path,slicename_t,label_mapping=label_mapping,device=device)

    hvg_s = adata_s.var[adata_s.var['highly_variable']].index
    hvg_t = adata_t.var[adata_t.var['highly_variable']].index
    hvg_intersection = list(set(hvg_s).intersection(set(hvg_t)))
    adata_s_hvg = adata_s[:, hvg_intersection].copy()
    adata_t_hvg = adata_t[:, hvg_intersection].copy()

    combined = np.concatenate([adata_s_hvg.X, adata_t_hvg.X], axis=0)
    mean = combined.mean(axis=0)
    std = combined.std(axis=0)
    std[std == 0] = 1

    adata_s_hvg.X = (adata_s_hvg.X - mean) / std
    adata_t_hvg.X = (adata_t_hvg.X - mean) / std


    adata_s.obsm['process'] = adata_s_hvg.X
    adata_t.obsm['process'] = adata_t_hvg.X


    graph_raw_s = graph_construction(adata_s, neighbor_nodes=6)
    graph_dict_s = {"indices": torch.tensor(graph_raw_s["indices"], dtype=torch.long).to(device)}

    graph_raw_t = graph_construction(adata_t, neighbor_nodes=6)
    graph_dict_t = {"indices": torch.tensor(graph_raw_t["indices"], dtype=torch.long).to(device)}
    input_dim = adata_s.obsm['process'].shape[1]

    # Initialize source domain encoder and classifier
    src_encoder = STransferEncoder(input_dim=input_dim).to(device)
    src_classifier = STransferClassifier(input_dim=64).to(device)

    # Initialize target domain encoder and domain discriminator (critic)
    tgt_encoder = STransferEncoder(input_dim=input_dim).to(device)
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

color_mapping = {
    0: "#5698D3",
    1: "#C2A16C",
    2: "#6F6DAF",
    3: "#8B2A2A",
    4: "#A65B8D",
    5: "#E6C44D",
}
# # Create legend handles for plot legends
legend_handles = [
    mlines.Line2D([], [], color=color, marker='o', linestyle='None', markersize=10, label=str(label))
    for label, color in color_mapping.items()]
# Assign pixel coordinates to the adata object
adata_s.obs['x_pixel'] = adata_s.obsm['spatial'][:,0]
adata_s.obs['y_pixel'] = adata_s.obsm['spatial'][:,1]
# Create figure with two subplots: true labels vs predicted labels
adata_s.obs['true'] = labels_s.cpu()
adata_s.obs["color_true"] = adata_s.obs['true'].map(color_mapping)
fig, axes = plt.subplots(1, 2, figsize=(20, 15))
# --- First subplot: Ground truth labels --
ax1 = axes[0]
scatter1 = ax1.scatter(adata_s.obs["x_pixel"], adata_s.obs["y_pixel"],
                       c=adata_s.obs["color_true"],
                       s=(900000 / adata_s.shape[0]) * 0.6,
                       alpha=0.8, edgecolor='black', linewidths=0.09)

ax1.set_xticks([])
ax1.set_yticks([])
ax1.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
# ax1.invert_yaxis()

for spine in ['top', 'right', 'left', 'bottom']:
    ax1.spines[spine].set_linewidth(0.8)
    ax1.spines[spine].set_color("black")

ax1.set_aspect('equal', 'box')
ax1.grid(False)

# --- Second subplot: Predicted labels ---
true_label1 = pred_labels_s
adata_s.obs['pred'] = true_label1
adata_s.obs["color"] = adata_s.obs['pred'].map(color_mapping)

ax2 = axes[1]
scatter2 = ax2.scatter(adata_s.obs["x_pixel"], adata_s.obs["y_pixel"],
                       c=adata_s.obs["color"],
                       s=(900000 / adata_s.shape[0]) * 0.6,
                       alpha=0.8, edgecolor='black', linewidths=0.09)

ax2.set_xticks([])
ax2.set_yticks([])
ax2.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
# ax2.invert_yaxis()
for spine in ['top', 'right', 'left', 'bottom']:
    ax2.spines[spine].set_linewidth(0.8)
    ax2.spines[spine].set_color("black")

ax2.set_aspect('equal', 'box')
ax2.grid(False)

# Add ARI and ACC score text to second subplot
ari_text = f"ARI: {ARI_s:.3f}"
acc_text = f"ACC: {ACC_s:.3f}"
ax2.text(0.25, 1.02, ari_text, ha='left', va='center', transform=ax2.transAxes, fontsize=20, color='black')
ax2.text(0.55, 1.02, acc_text, ha='left', va='center', transform=ax2.transAxes, fontsize=20, color='black')

ax1.invert_yaxis()
ax2.invert_yaxis()
plt.tight_layout()

source_clust_plot = os.path.join(params.model_root, params.source_clust_plot + ".png")
plt.savefig(source_clust_plot, dpi=300)
plt.close()
plt.show()

# --- Plot only the true labels as a standalone image ---
fig, ax = plt.subplots(figsize=(10, 10))
scatter = ax.scatter(adata_s.obs["y_pixel"], adata_s.obs["x_pixel"],
                     c=adata_s.obs["color_true"],
                     s=(900000 / adata_s.shape[0]) * 0.6,
                     alpha=0.8, edgecolor='black', linewidths=0.09)

ax.set_xticks([])
ax.set_yticks([])
ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
for spine in ['top', 'right', 'left', 'bottom']:
    ax.spines[spine].set_linewidth(0.8)
    ax.spines[spine].set_color("black")
ax.set_aspect('equal', 'box')
ax.grid(False)
ax.invert_yaxis()
plt.tight_layout()

source_clust_plot_true = os.path.join(params.model_root, params.source_clust_plot + "_true.png")
plt.savefig(source_clust_plot_true, dpi=300)
plt.close()
plt.show()

adata_s.obs['pred'] = pred_labels_s
adata_s.obs["color"] = adata_s.obs['pred'].map(color_mapping)

# --- Plot only the predicted labels as a standalone image ---
fig, ax = plt.subplots(figsize=(10, 10))
scatter = ax.scatter(adata_s.obs["y_pixel"], adata_s.obs["x_pixel"],
                     c=adata_s.obs["color"],
                     s=(900000 / adata_s.shape[0]) * 0.6,
                     alpha=0.8, edgecolor='black', linewidths=0.09)

ax.set_xticks([])
ax.set_yticks([])
ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
for spine in ['top', 'right', 'left', 'bottom']:
    ax.spines[spine].set_linewidth(0.8)
    ax.spines[spine].set_color("black")
ax.set_aspect('equal', 'box')
ax.grid(False)
ax.invert_yaxis()


ari_text = f"ARI: {ARI_s:.3f}"
acc_text = f"ACC: {ACC_s:.3f}"
ax.text(0.25, 1.02, ari_text, ha='left', va='center', transform=ax.transAxes, fontsize=20, color='black')
ax.text(0.55, 1.02, acc_text, ha='left', va='center', transform=ax.transAxes, fontsize=20, color='black')


plt.tight_layout()

source_clust_plot_pred = os.path.join(params.model_root, params.source_clust_plot + "_pred.png")
plt.savefig(source_clust_plot_pred, dpi=300)
plt.close()
plt.show()

# ############################################
# # -------- Plot for target dataset --------
############################################
# Assign spatial coordinates for target dataset
adata_t.obs['x_pixel'] = adata_t.obsm['spatial'][:,0]
adata_t.obs['y_pixel'] = adata_t.obsm['spatial'][:,1]


adata_t.obs['true'] = labels_t.cpu()
adata_t.obs["color_true"] = adata_t.obs['true'].map(color_mapping)
fig, axes = plt.subplots(1, 2, figsize=(20, 15))
ax1 = axes[0]

scatter1 = ax1.scatter(adata_t.obs["x_pixel"], adata_t.obs["y_pixel"], c=adata_t.obs["color_true"],
                       s=(900000 / adata_t.shape[0]) * 0.6, alpha=0.8, edgecolor='black', linewidths=0.09)

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
scatter2 = ax2.scatter(adata_t.obs["x_pixel"], adata_t.obs["y_pixel"], c=adata_t.obs["color"],
                       s=(900000 / adata_t.shape[0]) * 0.6, alpha=0.8, edgecolor='black', linewidths=0.09)

ax2.set_xticks([])
ax2.set_yticks([])
ax2.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)


for spine in ['top', 'right', 'left', 'bottom']:
    ax2.spines[spine].set_linewidth(0.8)
    ax2.spines[spine].set_color("black")

ax2.set_aspect('equal', 'box')
ax2.grid(False)


ari_text = f"ARI: {ARI_t:.3f}"
acc_text = f"ACC: {ACC_t:.3f}"
ax2.text(0.25, 1.02, ari_text, ha='left', va='center', transform=ax2.transAxes, fontsize=20, color='black')
ax2.text(0.55, 1.02, acc_text, ha='left', va='center', transform=ax2.transAxes, fontsize=20, color='black')

plt.tight_layout()
ax1.invert_yaxis()
ax2.invert_yaxis()

target_clust_plot = os.path.join(params.model_root, params.target_clust_plot + ".png")
plt.savefig(target_clust_plot, dpi=300)
plt.close()
plt.show()

fig, ax = plt.subplots(figsize=(10, 10))
ax.scatter(adata_t.obs["y_pixel"], adata_t.obs["x_pixel"],
           c=adata_t.obs["color_true"],
           s=(900000 / adata_t.shape[0]) * 0.6,
           alpha=0.8, edgecolor='black', linewidths=0.09)

ax.set_xticks([])
ax.set_yticks([])
ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

for spine in ['top', 'right', 'left', 'bottom']:
    ax.spines[spine].set_linewidth(0.8)
    ax.spines[spine].set_color("black")

ax.set_aspect('equal', 'box')
ax.grid(False)
ax.invert_yaxis()

plt.tight_layout()
target_clust_plot_true = os.path.join(params.model_root, params.target_clust_plot + "_true.png")
plt.savefig(target_clust_plot_true, dpi=300)
plt.close()
plt.show()

adata_t.obs['pred'] = pred_labels_t
adata_t.obs["color"] = adata_t.obs['pred'].map(color_mapping)

fig, ax = plt.subplots(figsize=(10, 10))
ax.scatter(adata_t.obs["y_pixel"], adata_t.obs["x_pixel"],
           c=adata_t.obs["color"],
           s=(900000 / adata_t.shape[0]) * 0.6,
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
ax.invert_yaxis()

plt.tight_layout()
target_clust_plot_pred = os.path.join(params.model_root, params.target_clust_plot + "_pred.png")
plt.savefig(target_clust_plot_pred, dpi=300)
plt.close()
plt.show()


min_samples = 2
counts = pd.Series(pred_labels_t).value_counts()
small_classes = counts[counts < min_samples].index.tolist()

pred_labels_t_merged = pred_labels_t.copy()
for cls in small_classes:
    candidates = [c for c in counts.index if c not in small_classes]
    nearest_cls = min(candidates, key=lambda x: abs(x - cls))
    pred_labels_t_merged[pred_labels_t == cls] = nearest_cls

adata_t.obs['pred_labels'] = pred_labels_t_merged.astype(str)


# ====================calculate marker gene==============================
sc.tl.rank_genes_groups(adata_t,
    groupby='pred_labels',
    method='t-test',
    n_genes=100)


marker_genes_dict = {}
groups = sorted(adata_t.obs['pred_labels'].unique())

for group in groups:
    genes = adata_t.uns['rank_genes_groups']['names'][group][:1]  # top5 marker
    marker_genes_dict[group] = list(genes)

print("\n===== Marker genes for each cluster =====")
for group, genes in marker_genes_dict.items():
    print(f"Cluster {group}: {genes}")

for group, genes in marker_genes_dict.items():
    for gene in genes:

        if gene not in adata_t.var_names:
            print(f"⚠️warning: gene {gene} not in var_names, skip plotting.")
            continue

        gene_exp = adata_t[:, gene].X
        gene_exp = gene_exp.toarray().flatten() if hasattr(gene_exp, "toarray") else gene_exp.flatten()
        norm_exp = (gene_exp - gene_exp.min()) / (gene_exp.max() - gene_exp.min() + 1e-9)

        fig, ax = plt.subplots(figsize=(9, 9))

        sc_ = ax.scatter(
            adata_t.obs["y_pixel"],
            adata_t.obs["x_pixel"],
            c=norm_exp,
            cmap="inferno_r",
            s=(900000 / adata_t.shape[0]) * 0.6,
            alpha=0.85,
            edgecolor='black',
            linewidths=0.08
        )

        #  cluster + marker gene
        ax.set_title(f"Cluster {group} – Marker: {gene}", fontsize=18)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.tick_params(left=False, bottom=False)
        for spine in ['top', 'right', 'left', 'bottom']:
            ax.spines[spine].set_visible(False)
        ax.set_aspect('equal', 'box')
        ax.invert_yaxis()
        plt.tight_layout()

        cbar = plt.colorbar(sc_, ax=ax)
        cbar.set_label(f"{gene} expression (normalized)")

        save_path = os.path.join(params.model_root, f"{params.target_clust_plot}_{gene}_target.png")
        plt.savefig(save_path, dpi=300)
        print(f"✅ Saved: {save_path}")
        plt.show()



