"""Params for ADDA."""
import os

# params for source dataset
class ParamConfig:
    def __init__(self, slicename_s, slicename_t):
        self.slicename_s = slicename_s
        self.slicename_t = slicename_t

        self.model_root = os.path.join("params", "DLPFC", 'STransfer', f"{slicename_s}-{slicename_t}")
        self.train_loss_name = f"{slicename_s}_source_train_loss"
        self.train_loss_plot = f"{slicename_s}_source_train_loss_plot"
        self.train_acc_name = f"{slicename_s}_source_train_acc"
        self.train_acc_plot = f"{slicename_s}_source_train_acc_plot"

        self.target_d_loss_name = f"{slicename_t}_target_train_d_loss"
        self.target_g_loss_name = f"{slicename_t}_target_train_g_loss"
        self.target_acc_name = f"{slicename_t}_accuracy"
        self.target_loss_acc_plot = f"{slicename_t}_target_loss_accuracy_plot"

        self.src_encoder_name = f"ADDA-{slicename_s}-{slicename_t}-source-encoder-final.pt"
        self.src_classifier_name = f"ADDA-{slicename_s}-{slicename_t}-source-classifier-final.pt"
        self.tgt_encoder_name = f"ADDA-{slicename_s}-{slicename_t}-target-encoder-final.pt"
        self.critic_name = f"ADDA-{slicename_s}-{slicename_t}-critic-final.pt"

        self.src_encoder_path = os.path.join(self.model_root, self.src_encoder_name)
        self.src_classifier_path = os.path.join(self.model_root, self.src_classifier_name)
        self.tgt_encoder_path = os.path.join(self.model_root, self.tgt_encoder_name)
        self.critic_path = os.path.join(self.model_root, self.critic_name)

        self.source_pred_labels = f"{slicename_s}_source_pred_labels"
        self.target_pred_labels = f"{slicename_t}_target_pred_labels"
        self.source_clust_plot = f"{slicename_s}_source_clust"
        self.target_clust_plot = f"{slicename_t}_target_clust"





