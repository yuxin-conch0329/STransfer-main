import os
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim


import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, adjusted_rand_score

from utils import save_model

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
# train source domain
def train_src(encoder, classifier, adata, graph_dict, labels, params):
    """Train classifier for source domain."""
    encoder.train()
    classifier.train()

    optimizer = optim.Adam(list(encoder.parameters()) + list(classifier.parameters()), lr=1e-5, betas=(0.5, 0.9))
    criterion = nn.CrossEntropyLoss()

    list_cls_loss = []
    acc_source = []
    num_epochs = 2000
    for epoch in range(num_epochs):
        x = torch.tensor(adata.obsm['process'], dtype=torch.float32).to(device)
        edge = graph_dict["indices"]

        optimizer.zero_grad()
        z = encoder(x, edge, cache_name='source')
        preds = classifier(z)
        cls_loss = criterion(preds, labels)
        cls_loss.backward()

        optimizer.step()
        pred_labels = torch.argmax(preds, dim=1).cpu().numpy()
        true_labels = labels.cpu().numpy()
        acc = accuracy_score(true_labels, pred_labels)
        acc_source.append(acc)
        print('epoch {}: Classification on the source domain. acc is {:.4f}'.format(epoch + 1, acc))
        list_cls_loss.append(cls_loss.detach().cpu().numpy())

    loss_save_path = os.path.join(params.model_root, params.train_loss_name + ".csv")
    os.makedirs(params.model_root, exist_ok=True)
    df_loss = pd.DataFrame({'epoch': list(range(1, len(list_cls_loss) + 1)),'train_loss': list_cls_loss})
    df_loss.to_csv(loss_save_path, index=False)

    acc_source_save_path = os.path.join(params.model_root, params.train_acc_name + ".csv")
    os.makedirs(params.model_root, exist_ok=True)
    df_acc = pd.DataFrame({'epoch': list(range(1, len(acc_source) + 1)),'acc': acc_source})
    df_acc.to_csv(acc_source_save_path, index=False)

    # save final model
    save_model(encoder, params.src_encoder_name, params)
    save_model(classifier, params.src_classifier_name, params)

    plt.rcParams['font.family'] = 'Times New Roman'
    loss_fig_path = os.path.join(params.model_root, params.train_loss_plot+".png")
    acc_fig_path = os.path.join(params.model_root, params.train_acc_plot + ".png")
    loss = pd.read_csv(loss_save_path)
    acc_total = pd.read_csv(acc_source_save_path)
    epochs = loss['epoch']

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(4, 4))
    ax1.plot(epochs, loss['train_loss'], label='Discriminator')
    ax1.set_title("Training Loss for Cluster Classifier", fontsize=12)
    ax1.grid(False)
    tick_step = 500
    xticks = range(0, len(epochs) + 1, tick_step)
    ax1.set_xticks(xticks)
    ax2.plot(epochs, acc_total['acc'], label='Domain Discriminator Accuracy', color='green')
    ax2.set_title("Accuracy for Source Classifier", fontsize=12)
    ax2.grid(False)
    ax2.set_xticks(xticks)
    plt.tight_layout()
    plt.savefig(acc_fig_path, dpi=300)

    return encoder, classifier


# evaluate source domain
def eval_src(encoder, classifier, adata, graph_dict, labels):
    """Evaluate classifier for source domain."""
    with torch.no_grad():
        encoder.eval()
        classifier.eval()

        criterion = nn.CrossEntropyLoss()
        x = torch.tensor(adata.obsm['process'], dtype=torch.float32).to(device)
        edge = graph_dict["indices"]

        z = encoder(x, edge, cache_name='source')
        preds = classifier(z)
        pred_labels = torch.argmax(preds, dim=1).cpu().numpy()

        loss = criterion(preds, labels).item()
        true_labels = labels.cpu().numpy()
        acc = accuracy_score(true_labels, pred_labels)

        print("Avg Loss = {}, Avg Accuracy = {:2%}".format(loss, acc))


# train target domain
def train_tgt(src_encoder, tgt_encoder, critic, adata_s, graph_dict_s, adata_t, graph_dict_t, params):
    """Train encoder for target domain."""
    tgt_encoder.train()
    critic.train()

    criterion = nn.CrossEntropyLoss()
    optimizer_tgt = optim.Adam(tgt_encoder.parameters(), lr=1e-5, betas=(0.5, 0.9))
    optimizer_critic = optim.Adam(critic.parameters(), lr=1e-4, betas=(0.5, 0.9))

    list_d_loss = []
    list_g_loss = []
    acc_total = []
    num_epochs = 2000
    for epoch in range(num_epochs):
        x_s = torch.tensor(adata_s.obsm['process'], dtype=torch.float32).to(device)
        edge_s = graph_dict_s["indices"]

        x_t = torch.tensor(adata_t.obsm['process'], dtype=torch.float32).to(device)
        edge_t  = graph_dict_t ["indices"]

        # zero gradients for optimizer
        optimizer_critic.zero_grad()

        # extract and concat features
        feature_src = src_encoder(x_s, edge_s, cache_name='source')
        feature_tgt = tgt_encoder(x_t, edge_t, cache_name='target')
        feature_concat = torch.cat((feature_src, feature_tgt), 0).to(device)

        # predict on discriminator
        pred_concat = critic(feature_concat.detach())

        label_src = torch.ones(feature_src.size(0), dtype=torch.long, device=feature_src.device)
        label_tgt = torch.zeros(feature_tgt.size(0), dtype=torch.long, device=feature_tgt.device)
        label_concat = torch.cat((label_src, label_tgt), dim=0)

        # compute loss for critic
        loss_critic = criterion(pred_concat, label_concat)
        loss_critic.backward()
        optimizer_critic.step()

        pred_cls = torch.squeeze(pred_concat.max(1)[1])
        acc = (pred_cls == label_concat).float().mean()

        # zero gradients for optimizer
        optimizer_critic.zero_grad()
        optimizer_tgt.zero_grad()

        # extract and target features
        feature_tgt = tgt_encoder(x_t, edge_t, cache_name='target')
        # predict on discriminator
        pred_tgt = critic(feature_tgt)
        # prepare fake labels
        label_tgt_fake = torch.ones(feature_tgt.size(0), dtype=torch.long, device=feature_tgt.device)

        # compute loss for target encoder
        loss_tgt = criterion(pred_tgt, label_tgt_fake)
        loss_tgt.backward()
        # optimize target encoder
        optimizer_tgt.step()

        list_d_loss.append(loss_critic.detach().cpu().numpy())
        list_g_loss.append(loss_tgt.detach().cpu().numpy())
        acc_total.append(acc.detach().cpu().numpy())

        d_loss_save_path = os.path.join(params.model_root, params.target_d_loss_name + ".csv")
        os.makedirs(params.model_root, exist_ok=True)
        df_loss = pd.DataFrame({'epoch': list(range(1, len(list_d_loss) + 1)),'train_loss': list_d_loss})
        df_loss.to_csv(d_loss_save_path, index=False)

        g_loss_save_path = os.path.join(params.model_root, params.target_g_loss_name + ".csv")
        os.makedirs(params.model_root, exist_ok=True)
        df_loss_g = pd.DataFrame({'epoch': list(range(1, len(list_g_loss) + 1)),'train_loss': list_g_loss})
        df_loss_g.to_csv(g_loss_save_path, index=False)

        acc_line_plot = os.path.join(params.model_root, params.target_acc_name + ".csv")
        os.makedirs(params.model_root, exist_ok=True)
        df_acc_line_plot = pd.DataFrame({'epoch': list(range(1, len(acc_total) + 1)),'acc': acc_total})

        df_acc_line_plot.to_csv(acc_line_plot, index=False)

        print("Epoch [{}/{}]: d_loss={:.5f} g_loss={:.5f} acc={:.5f}".format(epoch + 1, num_epochs, loss_critic.item(), loss_tgt.item(), acc.item()))


    # save model parameters #
    save_model(critic, params.critic_name,params)
    save_model(tgt_encoder, params.tgt_encoder_name,params)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))  # 2行1列子图
    # loss
    ax1.plot(list_d_loss, label='d_loss')
    ax1.plot(list_g_loss, label='g_loss')
    ax1.set_title("Training Loss for Domain Adaption")
    ax1.legend()

    # accuracy
    ax2.plot(acc_total, label='Domain Discriminator Accuracy', color='green')
    ax2.set_title("Accuracy for Domain Discriminator")
    ax2.legend()

    plt.tight_layout()

    target_l_a = os.path.join(params.model_root, params.target_loss_acc_plot + ".png")
    plt.savefig(target_l_a, dpi=300)
    plt.show()
    return tgt_encoder


# evaluate target domain
def eval_tgt(encoder, classifier, adata, graph_dict, labels):
    """Evaluation for target encoder by source classifier on target dataset."""
    encoder.eval()
    classifier.eval()
    criterion = nn.CrossEntropyLoss()

    # evaluate network
    x = torch.tensor(adata.obsm['process'], dtype=torch.float32).to(device)
    edge = graph_dict["indices"]
    labels = labels.to(device)

    z = encoder(x, edge, cache_name='target')
    preds = classifier(z)
    pred_labels = torch.argmax(preds, dim=1).cpu().numpy()
    loss = criterion(preds, labels).item()
    true_labels = labels.cpu().numpy()
    acc = accuracy_score(true_labels, pred_labels)
    ari = adjusted_rand_score(true_labels, pred_labels)
    print("Adjusted Rand Index (ARI):", ari)
    print("Avg Loss = {}, Avg Accuracy = {:2%}".format(loss, acc))

    return ari, pred_labels