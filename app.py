import numpy as np
import pandas as pd
import onnxruntime as ort
import streamlit as st
<<<<<<< HEAD
from sklearn.metrics import pairwise_distances
from scipy.sparse.linalg import eigsh
=======
#import backend
>>>>>>> b462f19ada40b01b1d390db6afbe3443a2a9018b


def diffusion_map(correlation_matrix, n_components=3, sigma=1.2):
    # Construct the affinity matrix directly from the correlation matrix
    affinity_matrix = np.exp(-pairwise_distances(correlation_matrix, metric='euclidean') ** 2 / (2. * sigma ** 2))

<<<<<<< HEAD
    # Normalize the affinity matrix
    degree_matrix = np.diag(np.sum(affinity_matrix, axis=1))
    diffusion_operator = np.dot(np.linalg.inv(degree_matrix), affinity_matrix)

    # Compute the top n_components eigenvalues and eigenvectors
    eigenvalues, eigenvectors = eigsh(diffusion_operator, k=n_components + 1, which='LM')


    diffusion_map = eigenvectors[:, 1:n_components + 1]

    return diffusion_map


st.title('Diffusion Enchanced BC-GCN')
tab1, tab2 = st.tabs(["Prediction Model", "Nerual Map"])

with tab1:
    st.write("This uses the diffusion map enabled BC-GCN model to estimate and the base BC-GCN model to classify Alzheimer's Disease. Please note that the classification model isn't always stable and the input must be a 112x112 corrolation matrix due to the preprocessing requiring external programs. **It is recommended that the example file be used.**")
    # Load the saved model

    pickle_file = "75.pkl"  # Path to the pickle file

    uploaded_file = st.file_uploader("Choose a file", type=["pkl"])
    checkbox_state = st.checkbox("Use Example File (Age: 75, State: AD)")

    # Check the state of the checkbox
    if checkbox_state:
        uploaded_file = pickle_file
    # Process the file based on its type
    if uploaded_file is not None:
        df = pd.read_pickle(uploaded_file)
        data_np_old = df.to_numpy()
        data_np = diffusion_map(data_np_old)

        input_data = np.expand_dims(data_np, axis=0)  # Add batch dimension (shape becomes [1, height, width])
        input_data = np.expand_dims(input_data, axis=0)  # Add channel dimension (shape becomes [1, 1, height, width])

        # Now convert input_data to float32 if it's not already
        input_data = input_data.astype(np.float32)

        # Load the ONNX model
        ort_session = ort.InferenceSession("model_dynamic_batch.onnx")
        input_feed = {'input.1': input_data}
        outputs = ort_session.run(None, input_feed)
        outputs = round(float(outputs[0][0][0]), 2)


        #dic = ['Mild Cognitive Impairment', 'Cognitively Normal', "Alzheimer's Disease"]


        st.markdown("<h1 style='text-align: center;'>Brain Age:</h1>", unsafe_allow_html=True)
        st.markdown(f"<h2 style='text-align: center;'>{outputs}</h2>", unsafe_allow_html=True)
        st.markdown("<h1 style='text-align: center;'>Brain Condition:</h1>", unsafe_allow_html=True)
        #st.markdown(f"<h2 style='text-align: center;'>{acc_value}</h2>", unsafe_allow_html=True)

with tab2:
    st.write("This is the neural map by using the weights of the BC-GCN model. Please select the Pial view to get a better visualization of the brain.")
    html_file = "map.html"
    with open(html_file, 'r') as file:
        html_content = file.read()

    # Display the HTML content in Streamlit
    st.components.v1.html(html_content, height=600)

=======
#output = backend.run(pickle_fil)


#st.title(output.item())
st.title("hello")
>>>>>>> b462f19ada40b01b1d390db6afbe3443a2a9018b
