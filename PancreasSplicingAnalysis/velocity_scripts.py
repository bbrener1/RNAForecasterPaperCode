import matplotlib.pyplot as plt 
from umap import UMAP
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist,squareform
import numpy as np


def delta_graph(origin,delta,highlight=[0,1],arrow_frequency=30,figsize=(10,8)):

    plt.figure(figsize=figsize)
    plt.scatter(*origin.T,color='red',s=1)

    for i,((x,y),(dx,dy)) in enumerate(zip(origin,delta)):
        if i%arrow_frequency == 0:
            plt.text(*origin[i],s=f"{i}")
            plt.arrow(x,y,dx,dy,color='green',head_width=.1,linewidth=.2)

    plt.show()

def mean_velo(tg,data):
    wsum = np.matmul(tg,data) 
    norm_const = np.sum(tg,axis=1)
    wmean = (wsum.T / norm_const).T
    velo = wmean - data
    return velo
    # length = np.sqrt(np.sum(np.power(velo,2),axis=1))
    # unitized = (velo.T / length).T
    # return unitized


def umap_velocity_via_transform(t0,t1):
    umap_model = UMAP(n_neighbors=15,min_dist=0.5, spread=1.0, n_components=2, negative_sample_rate=5, random_state=0,metric='cosine')
    u_t0 = umap_model.fit_transform(t0)
    u_t1 = umap_model.transform(t1)
    u_t_v = u_t1 - u_t0
    return u_t0,u_t1,u_t_v

def umap_velocity_via_joint(t0,t1):
    stacked = np.vstack([t0,t1])
    
    umap_model = UMAP(n_neighbors=15,min_dist=0.5, spread=1.0, n_components=2, negative_sample_rate=5, random_state=0,metric='cosine')
    u_t_joint = umap_model.fit_transform(stacked)
    
    u_t0 = u_t_joint[:t0.shape[0]]
    u_t1 = u_t_joint[t0.shape[0]:]
    
    u_t_v = u_t1 - u_t0
    return u_t0,u_t1,u_t_v


def umap_trajectory_joint(timepoints):
    if len(timepoints > 100):
        raise Exception("You want timepoints in a list, I think you might have passed a matrix, primary dimension > 100")
    stacked = np.vstack(timepoints)
    
    umap_model = UMAP(n_neighbors=15,min_dist=0.5, spread=1.0, n_components=2, negative_sample_rate=5, random_state=0,metric='cosine')
    u_t_joint = umap_model.fit_transform(stacked)
    
    running_totals = np.cumsum([t.shape[0] for t in timepoints])
    embedded_timepoints = [u_t_joint[beginning:end] for beginning,end in zip(running_totals[:-1],running_totals[1:])]
    
    return embedded_timepoints

def local_velocity_smoothness(velocities,knn,metric='cosine'):
    smoothness = []
    for neighborhood in knn:
        local_velocities = velocities[neighborhood]
        local_smoothness = np.mean(pdist(local_velocities,metric=metric))
        smoothness.append(local_smoothness)
    return smoothness


def extract_knn_from_adata(adata,k=None):

    samples = adata.shape[0]
    knn = []
    
    for i in range(samples):
        conn = np.array(adata.obsp['connectivities'][i].todense()).flatten()
        mask = conn > 0
        sort = np.argsort(conn[mask])
        indices = list(np.arange(samples)[mask][sort])
        knn.append(indices)
    
    clean = np.min([len(ragged) for ragged in knn])
    if k is None:
        k = clean
    else:
        if k > clean:
            print(f"WARNING: k connectivities aren't available everywhere. Setting to minimum k={clean}")
            k = clean
            
    knn = [ragged[-k:] for ragged in knn]
    return knn


def arrow_deltas(t1,t2):
    x,y = t1
    dx,dy = t2 - t1
    return (x,y,dx,dy)

def trajectory_series(trajectories,frequency=5):
    plt.figure(figsize=(30,30))
    plt.scatter(*trajectories[0].T,s=3)
    for t1,t2 in zip(trajectories[:-1],trajectories[1:]):
        for i in range(0,t1.shape[0],frequency): 
            plt.arrow(*arrow_deltas(t1[i],t2[i]),color='green',head_width=.05,linewidth=.2)
    plt.show()    