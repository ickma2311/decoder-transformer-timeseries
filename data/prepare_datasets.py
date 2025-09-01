"""
Data preparation script for time series comparison study.
Downloads and preprocesses HuggingFace time series datasets.
"""

import os
import pandas as pd
import numpy as np
from datasets import load_dataset
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

def download_monash_datasets():
    """Download key datasets from Monash Time Series Forecasting Archive."""
    print("Downloading Monash datasets...")
    
    # Popular univariate datasets for evaluation
    datasets_to_download = [
        "tourism",  # Monthly tourism data
        "traffic",  # Hourly traffic data  
        "electricity",  # Hourly electricity data
        "weather",  # Daily weather data
    ]
    
    monash_data = {}
    
    for dataset_name in datasets_to_download:
        try:
            print(f"  Loading {dataset_name}...")
            dataset = load_dataset("monash_tsf", dataset_name, trust_remote_code=True)
            monash_data[dataset_name] = dataset
            print(f"  ✓ {dataset_name}: {len(dataset['train'])} training series")
        except Exception as e:
            print(f"  ✗ Failed to load {dataset_name}: {e}")
            
    return monash_data

def download_additional_datasets():
    """Download additional time series datasets from HuggingFace."""
    print("\nDownloading additional datasets...")
    
    additional_data = {}
    
    try:
        # Financial time series
        print("  Loading ETT datasets (energy, temperature, oil)...")
        ett_h1 = load_dataset("ETTh1", trust_remote_code=True)
        additional_data["ett_h1"] = ett_h1
        print("  ✓ ETT-h1 loaded")
    except Exception as e:
        print(f"  ✗ Failed to load ETT datasets: {e}")
        
    return additional_data

def preprocess_monash_data(datasets):
    """Preprocess Monash datasets for model training."""
    print("\nPreprocessing datasets...")
    
    processed_data = {}
    
    for name, dataset in datasets.items():
        print(f"  Processing {name}...")
        
        try:
            # Extract train and test data
            train_data = dataset['train']
            test_data = dataset['test'] if 'test' in dataset else None
            
            # Convert to standardized format
            series_list = []
            
            for i, example in enumerate(train_data):
                # Extract time series values
                if 'target' in example:
                    values = example['target']
                elif 'series' in example:
                    values = example['series']
                else:
                    # Try to find numeric sequence
                    for key, val in example.items():
                        if isinstance(val, (list, np.ndarray)) and len(val) > 10:
                            values = val
                            break
                    else:
                        continue
                
                # Convert to numpy array and handle missing values
                values = np.array(values, dtype=float)
                values = values[~np.isnan(values)]  # Remove NaNs
                
                if len(values) >= 50:  # Minimum length requirement
                    series_list.append({
                        'series_id': f"{name}_{i}",
                        'values': values,
                        'length': len(values),
                        'frequency': getattr(example, 'frequency', 'unknown'),
                        'dataset': name
                    })
                    
                # Limit to first 20 series per dataset for initial testing
                if len(series_list) >= 20:
                    break
            
            processed_data[name] = series_list
            print(f"  ✓ {name}: {len(series_list)} series processed")
            
        except Exception as e:
            print(f"  ✗ Error processing {name}: {e}")
            
    return processed_data

def create_train_test_splits(processed_data, test_ratio=0.2):
    """Create train/test splits for each time series."""
    print("\nCreating train/test splits...")
    
    split_data = {}
    
    for dataset_name, series_list in processed_data.items():
        split_series = []
        
        for series in series_list:
            values = series['values']
            split_point = int(len(values) * (1 - test_ratio))
            
            # Ensure minimum lengths
            if split_point < 20 or len(values) - split_point < 5:
                continue
                
            train_values = values[:split_point]
            test_values = values[split_point:]
            
            split_series.append({
                **series,
                'train_values': train_values,
                'test_values': test_values,
                'train_length': len(train_values),
                'test_length': len(test_values)
            })
        
        split_data[dataset_name] = split_series
        print(f"  ✓ {dataset_name}: {len(split_series)} series with train/test splits")
    
    return split_data

def save_processed_data(split_data, output_dir="data"):
    """Save processed data to files."""
    print(f"\nSaving processed data to {output_dir}/...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save summary statistics
    summary = {
        'datasets': list(split_data.keys()),
        'total_series': sum(len(series) for series in split_data.values()),
        'series_per_dataset': {name: len(series) for name, series in split_data.items()}
    }
    
    # Save each dataset separately  
    for dataset_name, series_list in split_data.items():
        # Convert to DataFrame for easier handling
        df_data = []
        
        for series in series_list:
            df_data.append({
                'series_id': series['series_id'],
                'dataset': series['dataset'], 
                'train_length': series['train_length'],
                'test_length': series['test_length'],
                'frequency': series.get('frequency', 'unknown'),
                'train_values': series['train_values'].tolist(),
                'test_values': series['test_values'].tolist()
            })
        
        df = pd.DataFrame(df_data)
        output_path = os.path.join(output_dir, f"{dataset_name}_processed.csv")
        
        # Save metadata (everything except values)
        metadata_df = df.drop(['train_values', 'test_values'], axis=1)
        metadata_df.to_csv(output_path, index=False)
        
        # Save values separately as numpy arrays
        values_path = os.path.join(output_dir, f"{dataset_name}_values.npz")
        np.savez(values_path,
                train_values=[series['train_values'] for series in series_list],
                test_values=[series['test_values'] for series in series_list])
        
        print(f"  ✓ Saved {dataset_name}: {len(series_list)} series")
    
    # Save summary
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(os.path.join(output_dir, "dataset_summary.csv"), index=False)
    
    return summary

def main():
    """Main data preparation pipeline."""
    print("="*60)
    print("TIME SERIES DATA PREPARATION")
    print("="*60)
    
    # Step 1: Download datasets
    monash_data = download_monash_datasets()
    additional_data = download_additional_datasets()
    
    if not monash_data and not additional_data:
        print("No datasets downloaded successfully. Exiting.")
        return
    
    # Step 2: Preprocess data
    all_datasets = {**monash_data, **additional_data}
    processed_data = preprocess_monash_data(all_datasets)
    
    if not processed_data:
        print("No data processed successfully. Exiting.")
        return
    
    # Step 3: Create train/test splits
    split_data = create_train_test_splits(processed_data)
    
    # Step 4: Save processed data
    summary = save_processed_data(split_data)
    
    print("\n" + "="*60)
    print("DATA PREPARATION COMPLETE")
    print("="*60)
    print(f"Total datasets: {len(summary['datasets'])}")
    print(f"Total series: {summary['total_series']}")
    print("\nDatasets prepared:")
    for dataset, count in summary['series_per_dataset'].items():
        print(f"  {dataset}: {count} series")
    print("\nReady for model training and evaluation!")

if __name__ == "__main__":
    main()