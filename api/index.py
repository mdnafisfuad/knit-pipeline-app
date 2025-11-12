# /api/index.py
import os
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from ._utils.model_loader import load_all_models

# This is the main Flask app object that Vercel will run
app = Flask(__name__)
CORS(app)

# Load models ONCE when the serverless function starts up (a "cold start").
# The loaded models will be kept in memory for subsequent "warm" requests.
models = load_all_models()

@app.route('/api/models/info', methods=['GET'])
def get_models_info():
    """Provides the frontend with the structure of all loaded ML models."""
    all_info = {}
    for stage, model_api in models.items():
        categorical_options = {col: enc.classes_.tolist() for col, enc in model_api.label_encoders.items()}
        all_info[stage] = {
            'inputs': model_api.metadata['label_columns'],
            'outputs': model_api.metadata['feature_columns'],
            'categorical_options': categorical_options
        }
    return jsonify(all_info)

@app.route('/api/predict/<stage>', methods=['POST'])
def predict(stage):
    """Handles predictions for both algorithmic stages and ML models."""
    data = request.json
    try:
        # --- Algorithmic Stages ---
        if stage == 'order':
            target_gsm = float(data.get('req_gsm', 0))
            target_dia = float(data.get('req_dia', 0))
            weight_factor = float(data.get('weight_increase', 15.0)) / 100.0
            length_factor = float(data.get('length_increase', -20.0)) / 100.0
            
            sugg_gray_gsm = target_gsm / (1 + weight_factor)
            sugg_gray_dia = target_dia / (1 + length_factor)
            
            predictions = {'gray_gsm': round(sugg_gray_gsm, 2), 'gray_dia': round(sugg_gray_dia, 2)}
            return jsonify({'status': 'success', 'predictions': predictions})

        elif stage == 'dyeing':
            gray_gsm = float(data.get('produced_gray_gsm', 0))
            gray_dia = float(data.get('produced_gray_dia', 0))
            shade_percent = float(data.get('shade_percent', 0)) / 100.0
            enzyme_percent = float(data.get('enzyme_percent', 0)) / 100.0
            
            dyed_gsm = gray_gsm + (gray_gsm * shade_percent) - (gray_gsm * enzyme_percent)
            predictions = {'dyed_gsm': round(dyed_gsm, 2), 'dyed_dia': round(gray_dia, 2)}
            return jsonify({'status': 'success', 'predictions': predictions})
        
        # --- ML Model Stages ---
        if stage not in models:
            return jsonify({'status': 'error', 'message': f'Model for stage "{stage}" not loaded.'}), 404
        
        model = models[stage]
        predictions = model.generate(data)
        
        # Ensure all numeric outputs from ML models are also rounded
        for key, value in predictions.items():
            if isinstance(value, (int, float)):
                predictions[key] = round(value, 2)
                
        return jsonify({'status': 'success', 'predictions': predictions})

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

@app.route('/api/log', methods=['POST'])
def log_data():
    """Saves or updates a record in a CSV file in Vercel's temporary storage."""
    data = request.json
    # Vercel's serverless environment has a read-only filesystem, except for the /tmp directory.
    log_file = os.path.join('/tmp', 'process_history.csv')
    
    df = pd.DataFrame([data])
    all_columns = ['timestamp','batch_no','shade_percent','required_gsm','required_dia','construction','composition','sugg_gray_gsm','sugg_gray_dia','sugg_mc_dia','sugg_mc_gauge','sugg_yarn_count','sugg_stitch_length','sugg_tightness_factor','produced_gray_gsm','produced_gray_dia','enzyme_percent','sugg_dyed_gsm','sugg_dyed_dia','sugg_stenter_speed','sugg_stenter_temp','sugg_stenter_overfeed','sugg_stenter_set_dia','heat_set_gsm','heat_set_dia','sugg_compactor_speed','sugg_compactor_temp','sugg_compactor_overfeed','sugg_compactor_set_dia','finished_gsm','finished_dia']
    df['timestamp'] = pd.to_datetime('now').strftime('%Y-%m-%d %H:%M:%S')
    
    try:
        if os.path.exists(log_file):
            existing_df = pd.read_csv(log_file)
            if data['batch_no'] and data['batch_no'] in existing_df['batch_no'].values:
                existing_df.set_index('batch_no', inplace=True)
                update_series = pd.Series(data, name=data['batch_no']).combine_first(existing_df.loc[data['batch_no']])
                existing_df.update(pd.DataFrame(update_series).T)
                final_df = existing_df.reset_index()
            else:
                final_df = pd.concat([existing_df, df], ignore_index=True)
        else:
            final_df = df

        final_df.reindex(columns=all_columns).to_csv(log_file, index=False)
        return jsonify({'status': 'success'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'Log failed: {e}'}), 500

@app.route('/api/history', methods=['GET'])
def get_history():
    """Retrieves the history from the CSV file in Vercel's temporary storage."""
    log_file = os.path.join('/tmp', 'process_history.csv')
    if not os.path.exists(log_file):
        return jsonify([])
    df = pd.read_csv(log_file).fillna('').sort_values(by='timestamp', ascending=False)
    return jsonify(df.to_dict(orient='records'))