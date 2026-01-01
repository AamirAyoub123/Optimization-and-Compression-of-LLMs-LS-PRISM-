import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
import torch

def get_dynamic_rank(weight_matrix):
    # Compute SVD 
    device = weight_matrix.device
    U, S, Vh = torch.linalg.svd(weight_matrix.float(), full_matrices=False)
    
    # log transform as specified in study 
    log_S = torch.log(S).detach().cpu().numpy().reshape(-1, 1)
    

    if len(log_S) < 10 :
        r  = len(log_S)
    else:    
        # KMeans to identify dominant values 
        kmeans = KMeans(n_clusters=2, n_init=10).fit(log_S)
        dominant_cluster = np.argmax(kmeans.cluster_centers_)
        
        # Isolation Forest for significant singular values 
        iso_forest = IsolationForest(contamination=0.1,random_state=42)
        outliers = iso_forest.fit_predict(log_S)
        
        # rank selection: dominant cluster + outliers 
        mask = (kmeans.labels_ == dominant_cluster) | (outliers == -1)
        r = int(np.sum(mask))

        r = max(1,r)
    
    return r, U[:, :r].to(device), S[:r].to(device), Vh[:r, :].to(device)