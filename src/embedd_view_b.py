import torch
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load from .pth file to a tensor to CPU
quantized_embeddings = torch.load('quantized_embeddings_train.pth',map_location=torch.device('cpu'))
quantized_embeddings = torch.cat(quantized_embeddings)
quantized_embeddings = quantized_embeddings.cpu()

# Flatten tensor
quantized_embeddings_flat = quantized_embeddings.view(quantized_embeddings.size(0), -1)

# PCA
pca = PCA(n_components=2)
quantized_embeddings_pca = pca.fit_transform(quantized_embeddings_flat.numpy())

# quantized_embeddings plotting
plt.figure(figsize=(8, 6))
plt.scatter(quantized_embeddings_pca[:, 0], quantized_embeddings_pca[:, 1], s=10)
plt.title('Quantized Embeddings')
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.grid(True)
plt.show()
