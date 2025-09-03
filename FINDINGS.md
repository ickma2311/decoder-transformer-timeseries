# Project Findings: ts_comparison

This document lists code and logic findings after reviewing README.md, paper.md, and all Python modules.

## Critical Issues

- models/transformer_models.py (DecoderOnlyTransformer): Uses `nn.TransformerDecoder` with `tgt=x` and `memory=x` plus only `tgt_mask`. Cross-attention to `memory` is unmasked, so the model can attend to future positions via the memory pathway. This violates causal attention and undermines the core “decoder-only causal” claim.
- data/prepare_realworld_data.py (visualize_sample_data): Tries to read `NPZ['values']`, but the created NPZ files only contain `train_values` and `test_values`. This results in a KeyError when plotting.
- Real-world metadata missing: Evaluators expect `*_processed.csv` alongside `*_values.npz` (constructed via `dataset_path.replace('_values.npz', '_processed.csv')`). `prepare_realworld_data.py` only writes NPZs, so real-world evaluations will fail due to missing CSV.

## Functional Issues

- experiments/run_comparison.py (combine_results): Excludes key models from combined comparison and figures—specifically `Decoder_Only`, `Large_Transformer`, and `XGBoost`. This causes summaries/plots to omit models central to the paper’s claim.
- run_neural_fast.py: Loads `data/retail_sales_processed.csv`, which `prepare_realworld_data.py` does not create. This will raise FileNotFoundError.
- Zero-padding in real-world NPZs: Real-world NPZ arrays are padded with zeros to uniform length. Evaluators (`TraditionalModelEvaluator`, `TransformerModelEvaluator`) don’t trim padding when computing metrics (unlike moving-window evaluators that explicitly strip zeros). This can bias metrics if zero padding spills into the evaluated range.

## Documentation/Repo Mismatches

- README Quick Start references non-existent scripts: `experiments/run_transformer_eval.py` and `comprehensive_comparison.py`. Actual entry points are `experiments/run_comparison.py`, `run_neural_realworld.py`, and `run_moving_window_comprehensive.py`.
- Results filenames in README don’t match code outputs (e.g., README mentions `transformer_with_decoder_results.csv` while code writes `transformer_models_results.csv` or `transformer_results.csv`).
- models/transformer_models.py main summary omits `Decoder_Only` from the printed evaluation summary despite evaluating it.

## Minor/Stylistic Issues

- models/traditional_models_moving_window.py: Class named `MovingWindowXGBoost` actually uses LightGBM (`lgb.LGBMRegressor`). Naming is misleading.
- models/transformer_models.py: `DecoderOnlyTransformer.forward` has an unused `return_attention` parameter.
- models/traditional_models_simple.py (ProphetForecaster): Hardcoded `freq='H'` may be inappropriate for non-hourly series; consider parameterizing or inferring frequency.
- Forecast wrappers’ sliding-window assumption: Sequence update logic assumes `pred_len <= seq_len`. Works with current settings but is brittle if parameters change externally.
- PositionalEncoding mixes NumPy and Torch (`np.log` inside Torch context). Not incorrect, but could be standardized to pure Torch.

## Suggested Remediations (summary)

- Causal decoder: Replace `nn.TransformerDecoder` usage with masked self-attention only (e.g., `nn.TransformerEncoder` with a triangular causal mask) to avoid unmasked cross-attention.
- Real-world artifacts: When creating `retail_sales_values.npz` and `energy_consumption_values.npz`, also write matching `*_processed.csv` metadata, and update visualization to use `train_values`/`test_values` (strip zero padding for plots).
- Comparison pipeline: Update `combine_results()` to include `Decoder_Only`, `Large_Transformer`, and `XGBoost` (or auto-detect available columns). Align README with actual scripts and output filenames.
- Naming and params: Rename `MovingWindowXGBoost` → `MovingWindowLightGBM` or switch to xgboost; remove unused params; parameterize Prophet frequency.

