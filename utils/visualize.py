import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import cm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from torchvision import transforms
from .compute_confusion_matrix_visualize import compute_confusion_matrix
from .compute_features import compute_features
import torch.nn as nn
import umap.umap_ as umap


def fit_umap(tg_model, test_loader, testset, device, tb, mode, phases):
    tg_model.eval()
    tg_feature_model = nn.Sequential(*list(tg_model.children())[:-1])
    num_samples = len(testset.data)
    print(len(testset.data))
    num_features = tg_model.fc.in_features

    features = compute_features(tg_feature_model, test_loader, num_samples, num_features, device=device)

    # tsne = TSNE(2, verbose=1)
    trans = umap.UMAP(n_neighbors=5, random_state=42).fit(features[0:len(testset.data)])
    return trans


def visualize(trans, tg_model, test_loader, testset, device, tb, mode, phases):
    tg_model.eval()
    plt.style.use('default')
    tg_feature_model = nn.Sequential(*list(tg_model.children())[:-1])
    num_samples = len(testset.data) + len(testset.proto_sets_x)
    print(len(testset.data), len(testset.proto_sets_x))
    num_features = tg_model.fc.in_features

    features = compute_features(tg_feature_model, test_loader, num_samples, num_features, device=device)

    # tsne = TSNE(2, verbose=1)
    # trans = umap.UMAP(n_neighbors=5, random_state=42).fit(features[0:len(testset.data)])
    tsne_proj = trans.transform(features)
    print(tsne_proj.shape)
    # Plot those points as a scatter plot and label them based on the pred labels
    testset.return_type = True

    co_matrix, test_predictions, test_targets, markers = compute_confusion_matrix(tg_model, test_loader, device=device)
    testset.return_type = False
    marker_fig = [".", "*"]

    fig, ax = plt.subplots(figsize=(8, 8))
    num_categories = tg_model.fc.out_features
    cmap = cm.get_cmap('tab20')
    for lab in range(num_categories):
        indices_0 = markers == 0
        indices_1 = test_predictions == lab
        # print(len(indices_0),len(indices_1))
        indices = np.logical_and(indices_0, indices_1)
        mi = marker_fig[0]
        ax.scatter(tsne_proj[indices, 0], tsne_proj[indices, 1], c=np.array(cmap(lab)).reshape(1, 4), marker=mi,
                   label=str(lab), alpha=0.1)
    # for lab in range(num_categories):
    #     # indices = test_predictions==lab and markers == 1
    #     indices_0 = markers == 1
    #     indices_1 = test_predictions==lab 
    #     # print(len(indices_0),len(indices_1))
    #     indices = np.logical_and(indices_0, indices_1)
    #     mi = marker_fig[1]
    #     ax.scatter(tsne_proj[indices,0],tsne_proj[indices,1], c=np.array(cmap(lab)).reshape(1,4), marker = mi, label = 'Synthesized_'+str(lab) ,alpha=1)
    # ax.legend(fontsize='large', markerscale=2,loc='center left', bbox_to_anchor=(1, 1))
    # # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    # plt.savefig('/home/s.ayromlou/CVPR19_Incremental_Learning/cifar100-class-incremental/tsne/umap_'+tb+'_'+mode+'_'+str(phases)+'.png')
    for lab in range(num_categories):
        # indices = test_predictions==lab and markers == 1
        indices_0 = markers == 1
        indices_1 = test_targets == lab
        # print(len(indices_0),len(indices_1))
        indices = np.logical_and(indices_0, indices_1)
        mi = marker_fig[1]
        ax.scatter(tsne_proj[indices, 0], tsne_proj[indices, 1], c=np.array(cmap(lab)).reshape(1, 4), marker=mi,
                   label='Synthesized_' + str(lab), alpha=1)
    # lgd = ax.legend(fontsize='large', markerscale=2,loc='center left', bbox_to_anchor=(1, 0.5),ncol=len(df.columns))
    # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig(
        '/home/s.ayromlou/CVPR19_Incremental_Learning/cifar100-class-incremental/tsne/umap_label_' + tb + '_' + mode + '_' + str(
            phases) + '.png', bbox_inches='tight')
