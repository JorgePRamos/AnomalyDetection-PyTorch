import torch
import numpy as np
import matplotlib.pyplot as plt

# Load the quantized embeddings
quantized_embeddings = torch.load('quantized_embeddings_train.pth',map_location=torch.device('cpu'))

# Convert the quantized embeddings to a NumPy array
quantized_embeddings_np = torch.cat(quantized_embeddings).cpu().numpy()

# Choose the number of samples to visualize (e.g., first 5)
num_samples_to_visualize = 5
print(">> num exa: ",num_samples_to_visualize)
samples_to_visualize = quantized_embeddings_np[:num_samples_to_visualize]

# Plot each sample's embeddings as a grid
fig, axs = plt.subplots(num_samples_to_visualize, 1, figsize=(10, 8))

for i, sample in enumerate(samples_to_visualize):
    axs[i].imshow(sample.reshape(16, 16 * 50), cmap='viridis', aspect='auto')
    axs[i].set_xlabel('Embedding Dimension')
    axs[i].set_ylabel('Position in 16x16 Grid')

plt.tight_layout()
plt.show()
