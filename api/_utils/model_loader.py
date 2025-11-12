# /api/_utils/model_loader.py
import os
import pickle
import pandas as pd
import torch
import numpy as np
import requests  # For downloading
import zipfile   # For unzipping
import io        # To handle the downloaded file in memory
from .cvae_model import CVAE

# <<< --- 1. PASTE THE URLS YOU COPIED FROM GITHUB HERE --- >>>
MODEL_URLS = {
    "knitting": "https://github.com/YOUR_USERNAME/YOUR_REPO/releases/download/v1.0.0-models/knitting.zip",
    "stenter": "https://github.com/YOUR_USERNAME/YOUR_REPO/releases/download/v1.0.0-models/stenter.zip",
    "compactor": "https://github.com/YOUR_USERNAME/YOUR_REPO/releases/download/v1.0.0-models/compactor.zip",
    # Add a line for every model you have
}

class GenerativeModelAPI:
    # ... (This class definition is exactly the same as before) ...
    def __init__(self, model_path):
        self.model_dir = model_path; self.device = torch.device("cpu"); self.metadata = self.load_pickle('metadata.pkl'); self.label_encoders = self.load_pickle('label_encoders.pkl'); self.feature_transformer = self.load_pickle('feature_transformer.pkl'); self.label_scaler = self.load_pickle('label_scaler.pkl'); self.model = self.load_pytorch_model()
    def load_pickle(self, filename):
        with open(os.path.join(self.model_dir, filename), 'rb') as f: return pickle.load(f)
    def load_pytorch_model(self):
        model = CVAE(self.metadata['feature_dim'], self.metadata['label_dim'], self.metadata['latent_dim']); model.load_state_dict(torch.load(os.path.join(self.model_dir, 'cvae_model.pth'), map_location=self.device)); model.to(self.device); model.eval(); return model
    def generate(self, target_labels):
        target_df = pd.DataFrame([target_labels]); num_labels, cat_labels = self.metadata['numerical_labels'], self.metadata['categorical_labels']
        if num_labels: target_df[num_labels] = self.label_scaler.transform(target_df[num_labels])
        for col in cat_labels: target_df[col] = self.label_encoders[col].transform(target_df[col])
        labels_tensor = torch.tensor(target_df[self.metadata['label_columns']].values, dtype=torch.float32).to(self.device)
        with torch.no_grad(): z = torch.randn(1, self.metadata['latent_dim']).to(self.device); generated_scaled = self.model.decoder(torch.cat([z, labels_tensor], dim=1))
        generated_np = generated_scaled.cpu().numpy(); generated_features = self.feature_transformer.inverse_transform(generated_np)
        df = pd.DataFrame(generated_features, columns=self.metadata['feature_columns'])
        for col in self.metadata['categorical_features']:
            encoded = np.round(df[col]).astype(int); encoded = np.clip(encoded, 0, len(self.label_encoders[col].classes_) - 1); df[col] = self.label_encoders[col].inverse_transform(encoded)
        for col in self.metadata['numerical_features']: df[col] = np.round(df[col], 2)
        return df.to_dict(orient='records')[0]


# <<< --- 2. REPLACE THE OLD FUNCTION WITH THIS NEW DOWNLOADER LOGIC --- >>>
def load_all_models():
    """
    Downloads, unzips, and loads all ML models from external URLs
    into Vercel's temporary /tmp directory.
    """
    models = {}
    # Vercel's only writable directory
    target_dir = "/tmp/models"
    os.makedirs(target_dir, exist_ok=True)

    for stage_name, url in MODEL_URLS.items():
        try:
            stage_path = os.path.join(target_dir, stage_name)
            
            # Only download if it doesn't already exist in the warm function
            if not os.path.exists(stage_path):
                print(f"Downloading model for '{stage_name}' from {url}...")
                r = requests.get(url, stream=True)
                r.raise_for_status() # Will raise an error for bad status codes
                
                # Unzip the content in memory
                with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                    z.extractall(target_dir) # Extracts to /tmp/models/knitting/
                print(f"'{stage_name}' model unzipped to {stage_path}")
            
            # Now that the files are on disk, load the model
            print(f"Loading ML model from disk for stage: '{stage_name}'")
            models[stage_name] = GenerativeModelAPI(stage_path)

        except Exception as e:
            print(f"CRITICAL: Failed to download or load model for '{stage_name}'. Reason: {e}")
            
    return models
