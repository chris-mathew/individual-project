import numpy as np
import pandas as pd
import onnxruntime as ort
import streamlit as st
from sklearn.metrics import pairwise_distances
from scipy.sparse.linalg import eigsh


def diffusion_map(correlation_matrix, n_components=3, sigma=1.2):

    affinity_matrix = np.exp(-pairwise_distances(correlation_matrix, metric='euclidean') ** 2 / (2. * sigma ** 2))


    degree_matrix = np.diag(np.sum(affinity_matrix, axis=1))
    diffusion_operator = np.dot(np.linalg.inv(degree_matrix), affinity_matrix)


    eigenvalues, eigenvectors = eigsh(diffusion_operator, k=n_components + 1, which='LM')


    diffusion_map = eigenvectors[:, 1:n_components + 1]

    return diffusion_map



st.title('Diffusion Map Enchanced BC-GCN')
tab1, tab2 = st.tabs(["Prediction Model", "Neural Map"])

with tab1:
    st.write("This uses the diffusion map enabled BC-GCN model to estimate brain age and classify Alzheimer's Disease. Please note that the classification model isn't stable and the input must be a 112x112 corrolation matrix due to the preprocessing requiring external programs. **It is recommended that the example file be used.**")


    pickle_file = "75.9.pkl"  #  pickle file

    uploaded_file = st.file_uploader("Choose a file", type=["pkl"])
    checkbox_state = st.checkbox("Use Example File (Age: 75.9, State: AD, Subject ID (ADNI Dataset): 018_S_4733)")
    st.download_button(label="Download Sample Data", data=pickle_file, file_name='test_data.pkl',mime='application/octet-stream')


    if checkbox_state:
        uploaded_file = pickle_file

    if uploaded_file is not None:
        df = pd.read_pickle(uploaded_file)
        data_np_old = df.to_numpy()
        data_np = diffusion_map(data_np_old)

        input_data = np.expand_dims(data_np, axis=0)  #  [1, height, width]
        input_data = np.expand_dims(input_data, axis=0)  # [1, 1, height, width]
        input_data = input_data.astype(np.float32)

        input_data_acc = np.expand_dims(data_np_old, axis=0)  # (shape becomes [1, height, width])

        input_data_acc = input_data_acc.astype(np.float32)

        #ONNX model
        ort_session = ort.InferenceSession("model_dynamic_batch.onnx")
        input_feed = {'input.1': input_data}
        outputs = ort_session.run(None, input_feed)
        outputs = round(float(outputs[0][0][0]), 2)

        # ONNX model acc
        ort_session_acc = ort.InferenceSession("model_new_accuracy.onnx")
        input_feed_acc = {'input.1': input_data_acc}
        outputs_acc = ort_session_acc.run(None, input_feed_acc)
        outputs_acc = np.delete(outputs_acc[0][0],1)
        outputs_acc[0] = outputs_acc[0]/10
        outputs_acc = np.argmax(outputs_acc)


        dic = ['Mild Cognitive Impairment or Cognitively Normal', "Alzheimer's Disease"]

        st.markdown("<h1 style='text-align: center;'>Brain Age:</h1>", unsafe_allow_html=True)
        st.markdown(f"<h2 style='text-align: center;'>{outputs}</h2>", unsafe_allow_html=True)
        st.markdown("<h1 style='text-align: center;'>Brain Condition:</h1>", unsafe_allow_html=True)
        st.markdown(f"<h2 style='text-align: center;'>{dic[outputs_acc]}</h2>", unsafe_allow_html=True)



with tab2:
    st.write("This is the neural map by using the weights of the BC-GCN model. Please select the Pial view to get a better visualization of the brain.")
    html_file = "map.html"
    with open(html_file, 'r') as file:
        html_content = file.read()


    st.components.v1.html(html_content, height=600)