#!/usr/bin/env python3
"""
Debug DecoderOnly model initialization issue.
"""

import numpy as np
from models.transformer_models import DecoderOnlyForecaster

def debug_decoder():
    """Debug decoder only initialization."""
    
    print("Testing DecoderOnlyForecaster initialization...")
    
    try:
        # Test with the same parameters used in the main script
        model = DecoderOnlyForecaster(
            seq_len=20,
            pred_len=10,  # This might be the issue!
            d_model=48,
            nhead=6,
            num_layers=2,
            epochs=10,
            lr=0.001
        )
        print("✓ Model initialized successfully")
        
    except Exception as e:
        print(f"✗ Initialization failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Try without pred_len parameter
        print("\nTrying without pred_len parameter...")
        try:
            model = DecoderOnlyForecaster(
                seq_len=20,
                d_model=48,
                nhead=6,
                num_layers=2,
                epochs=10,
                lr=0.001
            )
            print("✓ Model initialized successfully without pred_len")
            
            # Test fitting
            print("Testing fit with sample data...")
            sample_data = np.random.randn(100)
            result = model.fit(sample_data)
            print(f"✓ Fit result: {result}")
            
        except Exception as e2:
            print(f"✗ Still failed: {e2}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    debug_decoder()