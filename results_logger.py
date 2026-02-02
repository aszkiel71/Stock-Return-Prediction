import json
import os
import datetime

RESULTS_FILE = 'model_results.json'

def log_results(model_name, scores, overall_acc, std_acc, description=""):
    """
    Logs model training results to a shared JSON file.

    Args:
        model_name (str): Name of the model.
        scores (list): List of k-fold accuracy scores.
        overall_acc (float): Mean accuracy.
        std_acc (float): Standard deviation of accuracy.
        description (str, optional): Additional details about the run (e.g., features used).
    """
    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Format the scores string as requested "Acc fold 1: ... etc"
    fold_str = ", ".join([f"Fold {i+1}: {s:.4f}" for i, s in enumerate(scores)])
    
    formatted_value = {
        "timestamp": timestamp,
        "description": description,
        "overall_accuracy": f"{overall_acc:.4f} (+- {std_acc:.4f})",
        "fold_details": fold_str
    }
    
    data = {}
    if os.path.exists(RESULTS_FILE):
        try:
            with open(RESULTS_FILE, 'r') as f:
                data = json.load(f)
        except json.JSONDecodeError:
            print(f"Warning: Could not decode {RESULTS_FILE}. Starting fresh.")
            data = {}
    
    # We append to a list for this model_name to keep history
    if model_name not in data:
        data[model_name] = []
        
    # If existing entry is just a string or dict (legacy/single entry), convert to list
    if not isinstance(data[model_name], list):
         data[model_name] = [data[model_name]]
         
    data[model_name].append(formatted_value)
        
    with open(RESULTS_FILE, 'w') as f:
        json.dump(data, f, indent=4)
        
    print(f"Results logged to {RESULTS_FILE} for {model_name}")
