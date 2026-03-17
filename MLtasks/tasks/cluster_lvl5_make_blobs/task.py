'''
make_blobs clustering implementation
1. generate the dataset
2. Train 
3. Verify and set the threshold 
4. Exit 
'''
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_blobs 
from sklearn.metrics import silhouette_score

def train_and_verify():
    # generate Synthetic Data (3 distinct clusters) for new dataset
    X, _ = make_blobs(n_samples=300, centers=3, n_features=2, cluster_std=0.6, random_state=42)
    X_tensor = torch.tensor(X, dtype=torch.float32)

    # define Trainable Centroids
    # initialize 3 centroids randomly from the data range
    centroids = nn.Parameter(torch.randn(3, 2))
    optimizer = optim.Adam([centroids], lr=0.1) # adam mentioned in class

    # training Loop (Minimizing Mean Squared Distance to nearest centroid)
    for epoch in range(100):
        optimizer.zero_grad()
        
        distances = torch.cdist(X_tensor, centroids)
        
        # Loss is the sum of distances to the closest centroid
        min_dist, _ = torch.min(distances, dim=1)
        loss = min_dist.mean()
        
        loss.backward()
        optimizer.step()

    # verification
    with torch.no_grad():
        distances = torch.cdist(X_tensor, centroids)
        cluster_labels = torch.argmin(distances, dim=1).numpy()
        
        score = silhouette_score(X, cluster_labels)
        
    print(f"Final Score: {score:.4f}")
    
    # success threshold:0.75 (indicating strong clustering) + exit
    if score >= 0.75:
        sys.exit(0)
    else:
        sys.exit(1)  

if __name__ == "__main__":
    train_and_verify()
