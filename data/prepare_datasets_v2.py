"""
Updated data preparation script using available HuggingFace datasets and synthetic data.
"""

import os
import pandas as pd
import numpy as np
from datasets import load_dataset
import warnings
warnings.filterwarnings("ignore")

def create_synthetic_datasets():
    """Create synthetic time series datasets for initial testing."""
    print("Creating synthetic datasets...")
    
    np.random.seed(42)
    synthetic_data = {}
    
    # Dataset 1: Trend + Seasonality + Noise
    print("  Creating trend_seasonal dataset...")
    series_list = []
    for i in range(20):
        t = np.arange(200)
        trend = 0.05 * t + np.random.normal(0, 0.5)
        seasonal = 5 * np.sin(2 * np.pi * t / 24) + 2 * np.cos(2 * np.pi * t / 12)
        noise = np.random.normal(0, 1, len(t))
        values = trend + seasonal + noise + 100  # Add baseline
        
        series_list.append({
            'series_id': f"trend_seasonal_{i}",
            'values': values,
            'length': len(values),
            'frequency': 'hourly',
            'dataset': 'trend_seasonal'
        })
    
    synthetic_data['trend_seasonal'] = series_list
    print(f"  ✓ trend_seasonal: {len(series_list)} series created")
    
    # Dataset 2: Multiple seasonality patterns
    print("  Creating multi_seasonal dataset...")
    series_list = []
    for i in range(20):
        t = np.arange(300)
        daily = 3 * np.sin(2 * np.pi * t / 24)
        weekly = 2 * np.sin(2 * np.pi * t / (24 * 7))
        monthly = 1.5 * np.sin(2 * np.pi * t / (24 * 30))
        noise = np.random.normal(0, 0.8, len(t))
        values = daily + weekly + monthly + noise + 50
        
        series_list.append({
            'series_id': f"multi_seasonal_{i}",
            'values': values,
            'length': len(values),
            'frequency': 'hourly',
            'dataset': 'multi_seasonal'
        })
    
    synthetic_data['multi_seasonal'] = series_list
    print(f"  ✓ multi_seasonal: {len(series_list)} series created")
    
    # Dataset 3: Random walks with drift
    print("  Creating random_walk dataset...")
    series_list = []
    for i in range(20):
        drift = np.random.uniform(-0.02, 0.05)
        innovations = np.random.normal(0, 1, 150)
        values = np.cumsum(innovations + drift) + 20
        
        series_list.append({
            'series_id': f"random_walk_{i}",
            'values': values,
            'length': len(values),
            'frequency': 'daily',
            'dataset': 'random_walk'
        })
    
    synthetic_data['random_walk'] = series_list
    print(f"  ✓ random_walk: {len(series_list)} series created")
    
    return synthetic_data

def try_download_real_datasets():
    """Try to download available real datasets from HuggingFace."""
    print("\nAttempting to download real datasets...")
    
    real_data = {}
    
    # Try some popular time series datasets
    dataset_attempts = [
        "autotrain-projects/sunspots",
        "huggingface/stock-prices", 
        "nielsr/electricity-transformer-et-dataset",
        "competition-math/test"
    ]
    
    for dataset_name in dataset_attempts:
        try:
            print(f"  Trying {dataset_name}...")
            dataset = load_dataset(dataset_name)
            
            # Basic processing - adapt based on actual structure
            if 'train' in dataset:
                real_data[dataset_name.split('/')[-1]] = dataset
                print(f"  ✓ {dataset_name}: loaded successfully")
                break  # Just get one for now
                
        except Exception as e:
            print(f"  ✗ Failed to load {dataset_name}: {str(e)[:100]}...")
            continue
    
    if not real_data:
        print("  No real datasets loaded - using synthetic data only")
    
    return real_data

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
    print("TIME SERIES DATA PREPARATION (Updated)")
    print("="*60)
    
    # Step 1: Create synthetic datasets (guaranteed to work)
    synthetic_data = create_synthetic_datasets()
    
    # Step 2: Try to download real datasets
    real_data = try_download_real_datasets()
    
    # Combine datasets
    all_datasets = {**synthetic_data, **real_data}
    
    if not all_datasets:
        print("No datasets created successfully. Exiting.")
        return
    
    # Step 3: Create train/test splits
    split_data = create_train_test_splits(all_datasets)
    
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