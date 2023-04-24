import numpy as np
import scanpy as sc
import seaborn as sns
import matplotlib.pyplot as plt
import time

def plot_tsne(X, label, prototype, label_text=None, save_path=None):

    start = time.time()

    adata = sc.AnnData(np.concatenate([X, prototype], axis=0))
    n_proto = len(prototype)
    label_proto = ["P_"+str(i) for i in range(n_proto)]

    sc.pp.neighbors(adata, n_neighbors=30)
    sc.pp.pca(adata, n_comps=min(50, min(X.shape[0], X.shape[1])-1))
    print("PCA Complete: {:d} min {:d} s".format(int((time.time()-start) // 60), int((time.time()-start) % 60)))
    sc.tl.tsne(adata, use_rep="X_pca")
    print("t-SNE Complete: {:d} min {:d} s".format(int((time.time()-start) // 60), int((time.time()-start) % 60)))

    
    plt.figure(figsize=(15,10))
    plt.clf()
    pal = sns.color_palette(n_colors = len(set(label)))
    pal_proto = sns.color_palette(n_colors = n_proto)

    if label_text is None:
        p = sns.scatterplot(x=adata.obsm["X_tsne"][:,0][:-n_proto], y=adata.obsm["X_tsne"][:,1][:-n_proto], hue=label, s=5, palette=pal, alpha=0.8)
    else:
        p = sns.scatterplot(x=adata.obsm["X_tsne"][:,0][:-n_proto], y=adata.obsm["X_tsne"][:,1][:-n_proto], hue=[label_text[i] for i in label], s=5, palette=pal, alpha=0.8)


    p.set(xlabel="tSNE1", ylabel="tSNE2")
    for i in range(n_proto):
        p.scatter(x=adata.obsm["X_tsne"][i-n_proto,0], y=adata.obsm["X_tsne"][i-n_proto,1], marker="p", s=150, color=pal_proto[i], label=label_proto[i])

        
    p.legend(bbox_to_anchor= (1,1))

    if save_path is not None:
        plt.savefig(save_path)

def plot_tsne_w_cont(X, label, prototype, contribution, class_text, label_text=None, save_path=None, s=5, pattern = "tab10"):

    start = time.time()

    adata = sc.AnnData(np.concatenate([X, prototype, contribution.reshape(-1, X.shape[1])], axis=0))
    n_proto = len(prototype)
    n_class = len(class_text)
    n_sample = contribution.shape[1]
    n_label = len(set(label))
    assert n_class == contribution.shape[0]
    label_proto = ["P_"+str(i) for i in range(n_proto)]

    sc.pp.neighbors(adata, n_neighbors=30)
    sc.pp.pca(adata, n_comps=min(50, min(X.shape[0], X.shape[1])-1))
    print("PCA Complete: {:d} min {:d} s".format(int((time.time()-start) // 60), int((time.time()-start) % 60)))
    sc.tl.tsne(adata, use_rep="X_pca")
    print("t-SNE Complete: {:d} min {:d} s".format(int((time.time()-start) // 60), int((time.time()-start) % 60)))

    plt.figure(figsize=(9,6))
    plt.clf()
    pal = sns.color_palette(pattern, n_colors = n_label)

    if label_text is None:
        p = sns.scatterplot(x=adata.obsm["X_tsne"][:,0][:-n_proto-n_class*n_sample], y=adata.obsm["X_tsne"][:,1][:-n_proto-n_class*n_sample], hue=label, s=s, palette=pal, alpha=0.8)
    else:
        p = sns.scatterplot(x=adata.obsm["X_tsne"][:,0][:-n_proto-n_class*n_sample], y=adata.obsm["X_tsne"][:,1][:-n_proto-n_class*n_sample], hue=[label_text[i] for i in label], s=s, palette=[pal[j] for j in range(len(label_text))], alpha=0.8)

    p.set(xlabel="tSNE1", ylabel="tSNE2")

    # pal_proto = sns.color_palette(n_colors = n_proto)
    # for i in range(n_proto):
    #     p.scatter(x=adata.obsm["X_tsne"][i-n_proto-n_class*n_sample,0], y=adata.obsm["X_tsne"][i-n_proto-n_class*n_sample,1], marker="p", s=250, color=pal_proto[i], label=label_proto[i])
    p.scatter(x=adata.obsm["X_tsne"][-n_proto-n_class*n_sample:-n_class*n_sample,0], y=adata.obsm["X_tsne"][-n_proto-n_class*n_sample:-n_class*n_sample,1], marker="p", s=250, color="black", label="P")

    # pal_class = sns.color_palette(n_colors = n_class)
    pal_class = ["black","darkred","mediumblue"]
    for j in range(n_class):
        i = n_class - 1 - j
        if s == 3:
            s_plus = 90
        else:
            s_plus = 150
        if i == n_class - 1:
            p.scatter(x=adata.obsm["X_tsne"][-(n_class-i)*n_sample:,0], y=adata.obsm["X_tsne"][-(n_class-i)*n_sample:,1], marker="+", s=s_plus, color=pal_class[i], label=class_text[i], alpha=0.8)
        else:
            p.scatter(x=adata.obsm["X_tsne"][-(n_class-i)*n_sample:-(n_class-i-1)*n_sample,0], y=adata.obsm["X_tsne"][-(n_class-i)*n_sample:-(n_class-i-1)*n_sample,1], marker="+", s=s_plus, color=pal_class[i], label=class_text[i], alpha=0.8)


    p.legend(bbox_to_anchor= (1,1))

    if save_path is not None:
        # plt.savefig(save_path)
        plt.savefig(save_path, bbox_extra_artists=(p,), bbox_inches='tight')

    plt.close()