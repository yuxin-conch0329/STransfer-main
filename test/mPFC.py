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

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'  # gpu

root_path = r"C:\E\JSU\BIO\file\STransfer\params\starmap"
pair_list = [
    ("BZ5", "BZ9"),
    ("BZ5", "BZ14"),
    ("BZ9", "BZ14")
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

    # === Evaluation on the target domains ===
    # Evaluate performance of domain-adapted target encoder
    print(">>> domain adaption <<<")
    ARI_t, pred_labels_t = eval_tgt(tgt_encoder, src_classifier, adata_t, graph_dict_t, labels_t)