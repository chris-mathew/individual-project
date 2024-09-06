import pandas as pd
import torch
from sklearn.metrics import pairwise_distances
from scipy.sparse.linalg import eigsh
import numpy as np
import streamlit as st

def diffusion_map(correlation_matrix, n_components=3, sigma=1.2):
    # Construct the affinity matrix directly from the correlation matrix
    affinity_matrix = np.exp(-pairwise_distances(correlation_matrix, metric='euclidean') ** 2 / (2. * sigma ** 2))

    # Normalize the affinity matrix
    degree_matrix = np.diag(np.sum(affinity_matrix, axis=1))
    diffusion_operator = np.dot(np.linalg.inv(degree_matrix), affinity_matrix)

    # Compute the top n_components eigenvalues and eigenvectors
    eigenvalues, eigenvectors = eigsh(diffusion_operator, k=n_components + 1, which='LM')

    # Skip the first eigenvector and use the rest
    # reducer = umap.UMAP(n_components=n_components)
    # diffusion_map = reducer.fit_transform(correlation_matrix)
    diffusion_map = eigenvectors[:, 1:n_components + 1]

    return diffusion_map


# Load the saved model
loaded_net = torch.load("entire_model.pt")
loaded_net.eval()  # Set to evaluation mode

# Load the pandas DataFrame from the pickle file
pickle_file = "73.4.pkl"  # Path to the pickle file
df = pd.read_pickle(pickle_file)

# Assuming your DataFrame has the correct structure expected by the model (for example, 2D or 3D array):
# Convert the DataFrame to a NumPy array
data_np = df.to_numpy()
data_np = diffusion_map(data_np)

# Convert the NumPy array to a PyTorch tensor
# Assuming that the model expects float64 tensors based on your earlier code
data_tensor = torch.tensor(data_np, dtype=torch.float64)


# If the input tensor needs to have a certain shape, make sure to reshape it appropriately.
# For example, if the model expects shape (batch_size, 1, num_regions, num_regions), you may need to unsqueeze the tensor:
data_tensor = data_tensor.unsqueeze(0).unsqueeze(0)
print(data_tensor.shape)

# Feed the tensor to the model
with torch.no_grad():  # Turn off gradient calculation since we're doing inference
    output = loaded_net(data_tensor)

# Print or process the output
print("Model output:", output.item())


st.title(output.item())
