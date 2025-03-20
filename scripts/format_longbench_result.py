import json
import pandas as pd

def format_longbench_result(json_path):
    # Read the JSON file
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Create a dictionary to store the results
    results = {}
    
    # Process each entry
    for key, value in data.items():
        if 'repobench-p' in key:
            key = key.replace('repobench-p', 'repobench_p')
        # Split the key into dataset name and method
        dataset, method = key.split('-')
        method = method.replace('.jsonl', '')
        
        # Initialize the dataset dict if not exists
        if dataset not in results:
            results[dataset] = {}
        
        # Store the value
        results[dataset][method] = value
    
    # Convert to DataFrame
    df = pd.DataFrame.from_dict(results, orient='index')
    
    # Reorder columns to match desired order
    desired_columns = ['full', 'minference', 'flex', 'xattn']
    df = df[desired_columns]
    
    # Sort by dataset name
    df = df.sort_index()
    
    # Calculate averages and prepend to DataFrame
    averages = df.mean()
    df.loc['average'] = averages
    
    # Round all values to 2 decimal places
    df = df.round(2)
    
    return df

if __name__ == "__main__":
    # Example usage
    result_df = format_longbench_result('eval/LongBench/pred/Meta-Llama-3.1-8B-Instruct/result.json')
    print("\nFormatted Results:")
    print(result_df.to_string())
    result_df.to_csv('eval/LongBench/pred/Meta-Llama-3.1-8B-Instruct/result.csv')