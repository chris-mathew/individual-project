#Initialisation
import streamlit as st
import pandas as pd
import torch

# Load the saved model
loaded_net = torch.load("C:/Users/chris/Desktop/ADData/SavedModels/entire_model_best_rmse_0.1234.pt")
loaded_net.eval()  # Set to evaluation mode

# Load the pandas DataFrame from the pickle file
pickle_file = r"C:\Users\chris\Desktop\ADData\CorrDATA\AD\002_S_5018\73.4.pkl"  # Path to the pickle file
df = pd.read_pickle(pickle_file)

# Assuming your DataFrame has the correct structure expected by the model (for example, 2D or 3D array):
# Convert the DataFrame to a NumPy array
data_np = df.to_numpy()

# Convert the NumPy array to a PyTorch tensor
# Assuming that the model expects float64 tensors based on your earlier code
data_tensor = torch.tensor(data_np, dtype=torch.float64)

# If the input tensor needs to have a certain shape, make sure to reshape it appropriately.
# For example, if the model expects shape (batch_size, 1, num_regions, num_regions), you may need to unsqueeze the tensor:
data_tensor = data_tensor.unsqueeze(1)  # Add the required dimension if needed

# Feed the tensor to the model
with torch.no_grad():  # Turn off gradient calculation since we're doing inference
    output = loaded_net(data_tensor)


#st.set_page_config(page_title='BI-RADS Score Determiner')

st.title(output)
