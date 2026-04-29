import os
import json
import pandas as pd
from datetime import datetime

def save_metrics(experiment_name, metrics, save_dir='results'):
    """
    Save metrics to a JSON file and append to a master CSV.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Add timestamp
    metrics['timestamp'] = datetime.now().isoformat()
    metrics['experiment'] = experiment_name
    
    # Save individual JSON
    json_path = os.path.join(save_dir, f'{experiment_name}_metrics.json')
    with open(json_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Saved metrics to {json_path}")
    
    # Append to master CSV
    csv_path = os.path.join(save_dir, 'all_experiments_metrics.csv')
    
    # Convert metrics to DataFrame (flatten if necessary)
    # For simplicity, we assume metrics is a flat dict or we flatten it
    flat_metrics = {}
    for k, v in metrics.items():
        if isinstance(v, dict):
            for sub_k, sub_v in v.items():
                flat_metrics[f"{k}_{sub_k}"] = sub_v
        else:
            flat_metrics[k] = v
            
    df = pd.DataFrame([flat_metrics])
    
    if os.path.exists(csv_path):
        df.to_csv(csv_path, mode='a', header=False, index=False)
    else:
        df.to_csv(csv_path, mode='w', header=True, index=False)
    print(f"Appended metrics to {csv_path}")
