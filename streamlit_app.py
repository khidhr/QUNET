import streamlit as st
from torchvision import transforms
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml
from pennylane import numpy as np
from scipy.ndimage import median_filter

class LowPramsEncoderSequential(nn.Module):
    def __init__(self, input_shape):
        super(LowPramsEncoderSequential, self).__init__()
        self.enc1 = nn.Conv2d(3, 128, kernel_size=3, padding='same')
        self.enc2 = nn.Conv2d(128, 256, kernel_size=5, padding='same')
        self.enc3 = nn.Conv2d(256, 128, kernel_size=3, padding='same')
        self.enc4 = nn.Conv2d(128, 64, kernel_size=5, padding='same')
        self.enc5 = nn.Conv2d(64, 32, kernel_size=3, padding='same')

        self.pool = nn.MaxPool2d(kernel_size=2, padding=0)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(32 * 12 * 12, 128)

    def forward(self, x):
    
        x1 = self.pool(nn.ReLU()(self.enc1(x)))
       
        x2 = self.pool(nn.ReLU()(self.enc2(x1)))
       
        x3 = self.pool(nn.ReLU()(self.enc3(x2)))
        
        x4 = self.pool(nn.ReLU()(self.enc4(x3)))
        
        x5 = self.pool(nn.ReLU()(self.enc5(x4)))
        
        x_flattened = self.flatten(x5)
     
        encoded = self.fc(x_flattened)
       
        return encoded, x1, x2, x3, x4, x5
    

n_qubits = 7

# Define the quantum device
dev = qml.device("default.qubit", wires=n_qubits)

# Define the quantum circuit
@qml.qnode(dev, interface='torch', diff_method="backprop")
def quantum_circuit(inputs, weights):
    qml.templates.AmplitudeEmbedding(inputs, wires=range(n_qubits), normalize=True)
    qml.templates.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
# Define the weight shapes
n_layers = 1
weight_shapes = {"weights": (n_layers, n_qubits, 3)}

# Define the quantum layer using qml.qnn.TorchLayer
quantum_layer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)

class LowParamsDecoderSequential(nn.Module):
    def __init__(self):
        super(LowParamsDecoderSequential, self).__init__()
        self.fc1 = nn.Linear(7, 128)
        self.fc2 = nn.Linear(128, 32 * 12 * 12)
        self.unflatten = nn.Unflatten(1, (32, 12, 12))
        
        # Define transposed convolutions with learnable parameters
        self.upconv1 = nn.ConvTranspose2d(32 + 32, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.upconv2 = nn.ConvTranspose2d(64 + 64, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.upconv3 = nn.ConvTranspose2d(128 + 128, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.upconv4 = nn.ConvTranspose2d(256 + 256, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.upconv5 = nn.ConvTranspose2d(64 + 128, 1, kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(self, x, x1, x2, x3, x4, x5):
        x = nn.ReLU()(self.fc1(x))
       
        x = nn.ReLU()(self.fc2(x))
        
        x = self.unflatten(x)
    
        x = nn.ReLU()(self.upconv1(torch.cat((x, x5), dim=1)))  # Concatenate and apply transposed convolution
      
        x = F.pad(x, (0, 1, 0, 1))
      
        x = nn.ReLU()(self.upconv2(torch.cat((x, x4), dim=1)))  # Concatenate and apply transposed convolution
       
        x = nn.ReLU()(self.upconv3(torch.cat((x, x3), dim=1)))  # Concatenate and apply transposed convolution
        
        x = nn.ReLU()(self.upconv4(torch.cat((x, x2), dim=1)))  # Concatenate and apply transposed convolution
        
        x = self.upconv5(torch.cat((x, x1), dim=1))  # Concatenate and apply transposed convolution
     
        x = torch.sigmoid(x) 
       
        return x
    
def build_lowparams_quantum_autoencoder_sequential():
    input_shape = (3, 400, 400)
    

    # Instantiate the encoder and decoder
    encoder = LowPramsEncoderSequential(input_shape=input_shape)
    decoder = LowParamsDecoderSequential()

    class Autoencoder(nn.Module):
        def __init__(self, encoder, decoder, quantum_layer):
            super(Autoencoder, self).__init__()
            self.encoder = encoder
            self.quantum_layer = quantum_layer
            self.decoder = decoder

        def forward(self, x):
            encoded, x1, x2, x3, x4, x5 = self.encoder(x)
            quantum_output = self.quantum_layer(encoded)
            decoded = self.decoder(quantum_output, x1, x2, x3, x4, x5)
            return decoded

    autoencoder = Autoencoder(encoder, decoder, quantum_layer)
    return autoencoder


LowParamsSegUnet = build_lowparams_quantum_autoencoder_sequential()

LowParamsSegUnet.load_state_dict(torch.load("lowParms_seg2.pth", map_location=torch.device('cpu')))
LowParamsSegUnet.eval()


transform = transforms.Compose([
    transforms.Resize((400, 400)),  # Resize to the input size of QUNET model
    transforms.ToTensor(),
])

############## mask post-processing ################
from scipy.ndimage import median_filter
def apply_median_filter(segmentation, size=3):
    return median_filter((segmentation >.6).astype(np.uint8), size=size)






def predict(image):
    image = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        output = LowParamsSegUnet(image)
    mask = output.squeeze().cpu().numpy() 
    return mask

st.title("Skin cancer Segmentation App using QUNET: a quantum enhanced UNET")


st.warning(
    "⚠️ This application is developed for research purposes only. "
    "The predicted segmentation masks are not intended for clinical or diagnostic use. "
    "Please consult a medical professional for accurate diagnosis and treatment."
)

st.write("Upload an image of skin cancer:")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.write("Segmenting...")
    
    mask = predict(image)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.image(image, caption="Uploaded Image", use_column_width=True)
    with col2:
        st.image(mask, caption="Raw Segmentation Mask", use_column_width=True)
    with col3:
        st.image(apply_median_filter(mask), caption="Processed Segmentation Mask", use_column_width=True)



