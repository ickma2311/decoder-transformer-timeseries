"""
Analyze the parameter count and model complexity of our custom transformer.
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from models.transformer_models import TimeSeriesTransformer, SimpleLSTMForecaster

def count_parameters(model):
    """Count total and trainable parameters in a model."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return total_params, trainable_params

def analyze_transformer_components():
    """Analyze parameter breakdown by component."""
    
    # Initialize model with our default parameters
    model = TimeSeriesTransformer(
        input_dim=1,
        d_model=64, 
        nhead=4,
        num_layers=2,
        seq_len=50,
        pred_len=10
    )
    
    print("="*60)
    print("CUSTOM TRANSFORMER PARAMETER ANALYSIS")
    print("="*60)
    
    # Total parameters
    total, trainable = count_parameters(model)
    print(f"Total Parameters: {total:,}")
    print(f"Trainable Parameters: {trainable:,}")
    print(f"Model Size: {total * 4 / 1024 / 1024:.2f} MB (float32)")
    print()
    
    # Component breakdown
    print("PARAMETER BREAKDOWN BY COMPONENT:")
    print("-" * 40)
    
    for name, param in model.named_parameters():
        params = param.numel()
        print(f"{name:35} | {params:>8,} | {param.shape}")
    
    print("-" * 40)
    
    # Component summaries
    input_proj_params = sum(p.numel() for n, p in model.named_parameters() 
                           if 'input_projection' in n)
    
    transformer_params = sum(p.numel() for n, p in model.named_parameters() 
                           if 'transformer_encoder' in n)
    
    output_proj_params = sum(p.numel() for n, p in model.named_parameters() 
                           if 'output_projection' in n)
    
    print(f"Input Projection:     {input_proj_params:>8,} params")
    print(f"Transformer Encoder:  {transformer_params:>8,} params")
    print(f"Output Projection:    {output_proj_params:>8,} params")
    print(f"Total:               {total:>8,} params")
    
    return model, total, trainable

def compare_with_lstm():
    """Compare transformer with our LSTM model."""
    
    print("\n" + "="*60)
    print("COMPARISON WITH LSTM MODEL")
    print("="*60)
    
    # LSTM model
    lstm_model = SimpleLSTMForecaster()._create_model()  # Get the internal model
    lstm_total, lstm_trainable = count_parameters(lstm_model)
    
    # Transformer model
    transformer_model = TimeSeriesTransformer()
    trans_total, trans_trainable = count_parameters(transformer_model)
    
    print(f"{'Model':<15} | {'Parameters':>12} | {'Size (MB)':>10}")
    print("-" * 45)
    print(f"{'LSTM':<15} | {lstm_total:>12,} | {lstm_total * 4 / 1024 / 1024:>9.2f}")
    print(f"{'Transformer':<15} | {trans_total:>12,} | {trans_total * 4 / 1024 / 1024:>9.2f}")
    print()
    
    ratio = trans_total / lstm_total
    print(f"Transformer has {ratio:.1f}x more parameters than LSTM")
    
    return lstm_total, trans_total

def theoretical_calculation():
    """Calculate parameters theoretically to verify."""
    
    print("\n" + "="*60)
    print("THEORETICAL PARAMETER CALCULATION")
    print("="*60)
    
    d_model = 64
    nhead = 4
    num_layers = 2
    pred_len = 10
    
    # Input projection: 1 -> 64
    input_proj = 1 * d_model + d_model  # weight + bias
    
    # Single transformer layer parameters
    # Multi-head attention
    d_k = d_model // nhead  # 16
    
    # Q, K, V projections: 64 -> 64 each
    qkv_proj = 3 * (d_model * d_model + d_model)  # 3 * (64*64 + 64)
    
    # Output projection: 64 -> 64  
    attn_out_proj = d_model * d_model + d_model
    
    # Layer norm 1: 64 parameters (scale + shift)
    ln1 = 2 * d_model
    
    # Feed forward: 64 -> 256 -> 64 (default expansion is 4x)
    ff_hidden = d_model * 4  # 256
    ff1 = d_model * ff_hidden + ff_hidden  # 64*256 + 256
    ff2 = ff_hidden * d_model + d_model    # 256*64 + 64
    
    # Layer norm 2: 64 parameters
    ln2 = 2 * d_model
    
    single_layer = qkv_proj + attn_out_proj + ln1 + ff1 + ff2 + ln2
    transformer_total = num_layers * single_layer
    
    # Output projection: 64 -> 32 -> 10
    out_proj1 = d_model * 32 + 32          # 64*32 + 32
    out_proj2 = 32 * pred_len + pred_len   # 32*10 + 10
    output_proj_total = out_proj1 + out_proj2
    
    theoretical_total = input_proj + transformer_total + output_proj_total
    
    print(f"Input Projection:     {input_proj:>8,}")
    print(f"Transformer Encoder:  {transformer_total:>8,}")
    print(f"  - Per layer:        {single_layer:>8,}")
    print(f"  - QKV projections:  {qkv_proj:>8,}")
    print(f"  - Attention out:    {attn_out_proj:>8,}")
    print(f"  - Feed forward:     {ff1 + ff2:>8,}")
    print(f"  - Layer norms:      {ln1 + ln2:>8,}")
    print(f"Output Projection:    {output_proj_total:>8,}")
    print(f"THEORETICAL TOTAL:    {theoretical_total:>8,}")
    
    return theoretical_total

def main():
    """Run complete parameter analysis."""
    
    # Actual parameter count
    model, actual_total, trainable = analyze_transformer_components()
    
    # Compare with LSTM
    lstm_params, trans_params = compare_with_lstm()
    
    # Theoretical calculation
    theoretical_total = theoretical_calculation()
    
    # Summary
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    print(f"Actual Parameters:      {actual_total:>8,}")
    print(f"Theoretical Parameters: {theoretical_total:>8,}")
    print(f"Match: {'✅ Yes' if actual_total == theoretical_total else '❌ No'}")
    print()
    print("MODEL COMPLEXITY COMPARISON:")
    print(f"• Our Transformer: {actual_total:,} params")
    print(f"• Our LSTM:        {lstm_params:,} params") 
    print(f"• Ratio:           {actual_total/lstm_params:.1f}x larger")
    print()
    print("CONTEXT:")
    print(f"• GPT-2 Small:     117M params ({117_000_000/actual_total:.0f}x larger)")
    print(f"• BERT Base:       110M params ({110_000_000/actual_total:.0f}x larger)")
    print(f"• Our model:       {actual_total/1000:.1f}K params (very lightweight!)")

if __name__ == "__main__":
    main()