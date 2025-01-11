import streamlit as st
import torch
from typing import List

# Define DNA helpers
alphabet = 'nacgt'  # We use lowercase alphabet, but input can be upper or lower case
dna2int = {a: i for a, i in zip(alphabet, range(5))}
int2dna = {i: a for a, i in zip(alphabet, range(5))}

class CpGPredictor(torch.nn.Module):
    ''' Simple model that uses an LSTM to count the number of CpGs in a sequence '''
    def __init__(self, input_size=5, hidden_size=64, num_layers=2, output_size=1):
        super(CpGPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = torch.nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.classifier = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.long()
        x = torch.nn.functional.one_hot(x, num_classes=5).float()
        lstm_out, _ = self.lstm(x)
        last_step_output = lstm_out[:, -1, :]
        logits = self.classifier(last_step_output)
        return logits

# Load the trained model (update the path to your saved model file if needed)
@st.cache_resource  # Cache the loaded model to avoid reloading on every interaction
def load_model():
    model = CpGPredictor(input_size=5, hidden_size=64, num_layers=2, output_size=1)
    model.load_state_dict(torch.load(r"C:\Users\APJ\OneDrive\Desktop\machine_task\cpg_predictor.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

# Prediction function
def predict_cpg_count(dna_sequence: str, model: CpGPredictor) -> int:
    # Ensure the input is in lowercase, so case doesn't matter
    dna_sequence = dna_sequence.lower()
    
    # Convert the sequence to integers using dna2int mapping
    int_sequence = [dna2int[char] for char in dna_sequence if char in dna2int]
    input_tensor = torch.tensor([int_sequence], dtype=torch.long)
    
    # Get the model's prediction without calculating gradients
    with torch.no_grad():
        prediction = model(input_tensor)
    
    # Round to nearest whole number (integer)
    return round(prediction.item())

# Streamlit App UI
st.title("CpG Predictor")
st.write("Input a DNA sequence, and this app will predict the count of CpGs (consecutive 'CG' pairs).")

# Input section
dna_sequence = st.text_input("Enter a DNA sequence (using characters: n, a, c, g, t):")

# Prediction and output
if dna_sequence:
    # Validate the input sequence (ensure only valid characters)
    valid_chars = set(alphabet)
    if all(char in valid_chars for char in dna_sequence.lower()):  # Check case-insensitive
        predicted_cpgs = predict_cpg_count(dna_sequence, model)
        st.success(f"Predicted CpG count: {predicted_cpgs}")
    else:
        st.error("Invalid DNA sequence. Please use only the characters: n, a, c, g, t.")
