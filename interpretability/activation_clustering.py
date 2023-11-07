import numpy as np
import matplotlib.pyplot as plt
import sklearn 
from sklearn.cluster import DBSCAN,KMeans
from sklearn.decomposition import NMF
from pathlib import Path
activations_cmap = "plasma"

def get_cmap(filepath):
    from matplotlib.colors import LinearSegmentedColormap
    cmap_list = ["red", "white", "green"]
    if str(filepath).endswith("rise.png"):
        cmap_list = ["white", "green"]
    activations_cmap = LinearSegmentedColormap.from_list(
            "RdWhGn", cmap_list
        )
    return activations_cmap

def get_nmf(activations,k):
    n,h,w=activations.shape
    activations = activations.reshape((n,h*w))
    # print(activations.shape)
    decomposition = NMF(n_components=k, init='random',max_iter=500)
    decomposition.fit_transform(activations)
    components = decomposition.components_.T.reshape((h,w,-1))
    # print(components.shape)
    components_importance = np.abs(components).mean(axis=(0,1))
    components_importance /= components_importance.sum()

    indices = np.flip(np.argsort(components_importance))
    components_importance = components_importance[indices]
    components= components[:,:,indices]
    return components,components_importance

def plot_activation_nmf(activations,filepath,k_col_row,fontsize_title=20):
    activations_cmap = get_cmap(filepath)

    k=k_col_row**2    
    activations = np.abs(activations)

    components,components_importance = get_nmf(activations,k)
    
    f,axis = plt.subplots(k_col_row,k_col_row,dpi=200,figsize=(k_col_row*4,k_col_row*4), squeeze=False, constrained_layout=True)
    axis = axis.reshape((-1))
    for i,ax in enumerate(axis):
        im=ax.imshow(components[:,:,i],cmap=activations_cmap)
        ax.set_axis_off()
        ax.set_title(f"Importance: {components_importance[i]*100:.1f}%",fontsize=fontsize_title)
        # f.colorbar(im, ax=ax, shrink=0.75)
    f.colorbar(im, ax=axis.ravel().tolist(), location='right')
    # plt.tight_layout()
    plt.savefig(filepath)

def get_clusters(activations,k):
    # print(activations.shape,"k",k)
    n,h,w=activations.shape
    activations = activations.reshape((n,h*w))
    clustering = KMeans(n_clusters=k).fit(activations)
    centers = clustering.cluster_centers_.T.reshape(h,w,-1)
    # print("centers",centers.shape)

    ## Calculate cluster importance based on relative magnitude
    activation_to_cluster = np.identity(k)[clustering.labels_]
    activation_importance = np.abs(activations).mean(axis=(1))
    
    cluster_importance = activation_importance.dot(activation_to_cluster)
    cluster_importance /=cluster_importance.sum()
    
    # determine cluster size 
    cluster_counts = np.bincount(clustering.labels_)
    cluster_percentages = cluster_counts/n

    # Sort clusters by importance
    indices = np.flip(np.argsort(cluster_importance))
    cluster_importance = cluster_importance[indices]
    cluster_percentages = cluster_percentages[indices]
    centers = centers[:,:,indices]
    return centers,cluster_importance,cluster_percentages

def plot_activation_clusters(activations,filepath,k_col_row,fontsize_title=20):
    activations_cmap = get_cmap(filepath)

    # clustering = DBSCAN(eps=3,min_samples=c//5).fit(cluster_activations)
    k=k_col_row**2    
    centers,cluster_importance,cluster_percentages = get_clusters(activations,k)
    mi,ma = centers.min(),centers.max()
    f,axis = plt.subplots(k_col_row,k_col_row,dpi=200,figsize=(k_col_row*4,k_col_row*4), squeeze=False, constrained_layout=True)
    axis = axis.reshape((-1))
    for i,ax in enumerate(axis):
        im=ax.imshow(centers[:,:,i],cmap=activations_cmap,vmin=mi,vmax=ma)
        ax.set_axis_off()
        ax.set_title(f"Importance: {cluster_importance[i]*100:.1f}%\n Size:{cluster_percentages[i]*100:.1f}%",fontsize=fontsize_title)
        # f.colorbar(im, ax=ax, shrink=0.75)
    f.colorbar(im, ax=axis.ravel().tolist(), location='right')
    # plt.tight_layout()        
    # f.colorbar(im, ax=axis.tolist())
    
    
    plt.savefig(filepath)

if __name__ == '__main__':
    filenames = [Path("0_UNet_class0_activations.npy"),
                Path("1_UNet_class1_activations.npy")]
    # filename = 
    for filename in filenames:
        print(f"Processing {filename}..")
        activations = np.load(filename)
        
        filepath = f"{filename.stem}_kmeans.png"
        plot_activation_clusters(activations,filepath)
        
        filepath_nmf = f"{filename.stem}_nmf.png"
        plot_activation_nmf(activations,filepath_nmf,3)



