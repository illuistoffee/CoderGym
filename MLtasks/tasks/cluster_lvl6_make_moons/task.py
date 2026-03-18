'''
make_moons clustering implementation. makes a clustered dataset with two crescents
1. generate the dataset
2. Train 
3. Verify and set the threshold 
4. Exit 
'''
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_moons 
from sklearn.metrics import silhouette_score

def train_and_verify():
    # generate Synthetic Data (2 crescent clusters) for new dataset
    X, _ = make_moons(n_samples=300, noise=0.07, random_state=42)
    X_tensor = torch.tensor(X, dtype=torch.float32)

    # define Trainable Centroids
    # already has 2 centroids
    centroids = nn.Parameter(torch.randn(2, 2))
    optimizer = optim.Adam([centroids], lr=0.1) # adam mentioned in class

    # training Loop (Minimizing Mean Squared Distance to nearest centroid)
    for epoch in range(150):
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
    
    # success threshold:0.40 + exit
    if score >= 0.40:
        sys.exit(0)
    else:
        sys.exit(1)  

if __name__ == "__main__":
    train_and_verify()
