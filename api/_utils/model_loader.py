# /api/_utils/model_loader.py
import os
import pickle
import pandas as pd
import torch
import numpy as np
from .cvae_model import CVAE # Relative import from the same _utils folder

class GenerativeModelAPI:
    """
    A class to encapsulate a single trained CVAE model and its assets.
    """
    def __init__(self, model_path):
        self.model_dir = model_path
        self.device = torch.device("cpu") # Vercel runs on CPU, so we force this
        
        # Load all necessary assets for the model
        self.metadata = self.load_pickle('metadata.pkl')
        self.label_encoders = self.load_pickle('label_encoders.pkl')
        self.feature_transformer = self.load_pickle('feature_transformer.pkl')
        self.label_scaler = self.load_pickle('label_scaler.pkl')
        self.model = self.load_pytorch_model()
        
    def load_pickle(self, filename):
        """Helper to load a pickle file from the model directory."""
        with open(os.path.join(self.model_dir, filename), 'rb') as f:
            return pickle.load(f)

    def load_pytorch_model(self):
        """Initializes the CVAE model structure and loads the trained weights."""
        model = CVAE(
            self.metadata['feature_dim'],
            self.metadata['label_dim'],
            self.metadata['latent_dim']
        )
        # map_location ensures the model loads correctly on a CPU-only environment
        model.load_state_dict(torch.load(os.path.join(self.model_dir, 'cvae_model.pth'), map_location=self.device))
        model.to(self.device)
        model.eval()
        return model

    def generate(self, target_labels):
        """
        The main prediction function. Takes inputs, preprocesses them, runs the model,
        and post-processes the output.
        """
        # Pre-processing
        target_df = pd.DataFrame([target_labels])
        num_labels, cat_labels = self.metadata['numerical_labels'], self.metadata['categorical_labels']
        if num_labels:
            target_df[num_labels] = self.label_scaler.transform(target_df[num_labels])
        for col in cat_labels:
            target_df[col] = self.label_encoders[col].transform(target_df[col])
        
        labels_tensor = torch.tensor(target_df[self.metadata['label_columns']].values, dtype=torch.float32).to(self.device)
        
        # Model Inference
        with torch.no_grad():
            z = torch.randn(1, self.metadata['latent_dim']).to(self.device)
            generated_scaled = self.model.decoder(torch.cat([z, labels_tensor], dim=1))
        
        # Post-processing
        generated_np = generated_scaled.cpu().numpy()
        generated_features = self.feature_transformer.inverse_transform(generated_np)
        
        df = pd.DataFrame(generated_features, columns=self.metadata['feature_columns'])
        
        for col in self.metadata['categorical_features']:
            encoded = np.round(df[col]).astype(int)
            encoded = np.clip(encoded, 0, len(self.label_encoders[col].classes_) - 1)
            df[col] = self.label_encoders[col].inverse_transform(encoded)
            
        for col in self.metadata['numerical_features']:
            df[col] = np.round(df[col], 2)
            
        return df.to_dict(orient='records')[0]

def load_all_models():
    """
    Discovers and loads all valid ML models from the /api/models directory.
    This function is called once when the serverless function starts.
    """
    # Get the absolute path to the 'models' directory next to this file
    models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    models = {}
    
    if not os.path.exists(models_dir):
        print(f"Warning: Models directory not found at {models_dir}")
        return models
        
    for stage_name in os.listdir(models_dir):
        stage_path = os.path.join(models_dir, stage_name)
        if os.path.isdir(stage_path):
            # Skip stages handled by algorithms
            if stage_name in ['order', 'dyeing']:
                continue
            try:
                print(f"Loading ML model for stage: '{stage_name}'")
                models[stage_name] = GenerativeModelAPI(stage_path)
            except Exception as e:
                print(f"Failed to load model from '{stage_path}': {e}")
    return models