import warnings

from params import ParamConfig
from utils import *
from modules import *
from train import *

# environment
warnings.filterwarnings('ignore')
random_seed = 2023
fix_seed(random_seed)
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# global setting
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'  # gpu
label_mapping = {'L1': 1, 'L2': 2, 'L3': 3, 'L4': 4, 'L5': 5, 'L6': 6, 'WM': 0}
# data_dir = "C:/Users/Administrator/Desktop/Yu Xin/7.18/data"
# data_dir = "/root/STransfer/data/"
# data_dir = "/home/yuxin/modle/SPAGCN/SpaGCN-master/tutorial/"
data_dir =r"C:\Users\YUXIN\Desktop\new\new\data"
pair_list = [
    ("151507", "151508"),
    # ("151509", "151510"),
    # ("151669", "151670"),
    # ("151671", "151672"),
    # ("151673", "151674"),
    # ("151675", "151676")
]

# Iterate over each pair in the slice list (source and target domains)
# for a, b in pair_list:
for slicename_s, slicename_t in pair_list:
# for slicename_s, slicename_t in [(a, b)]:
    print(f" processing：source={slicename_s} | target={slicename_t}")
    # Initialize parameter configuration
    n_top_genes = 1000
    params = ParamConfig(slicename_s, slicename_t)
    adata_s, data_s, labels_s, graph_dict_s = preprocess_slice_DLPFC(slicename_s, data_dir, device, label_mapping,n_top_genes)
    adata_t, data_t, labels_t, graph_dict_t = preprocess_slice_DLPFC(slicename_t, data_dir, device, label_mapping,n_top_genes)

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

    # === Evaluation on the target domains ===
    # Evaluate performance of domain-adapted target encoder
    print(">>> domain adaption <<<")
    ARI_t, pred_labels_t = eval_tgt(tgt_encoder, src_classifier, adata_t, graph_dict_t, labels_t)


