# Multivariate (2D) Time Series Forecasting — Direction & Plan

This document outlines datasets, benchmarks, modeling approaches, and concrete repo changes to pivot toward multivariate (2D) time series forecasting where multiple series (e.g., products, sensors, clients) are modeled jointly to learn cross-item relations.

## Why Multivariate
- Cross-item relations: Learn shared seasonality, substitution/complementarity, and promotions’ spillovers across items/stores.
- Data efficiency: One global model across items improves generalization and reduces per-series tuning.
- Strong empirical results: On large panels (e.g., M5, Electricity, ETT), global ML/deep models typically outperform per-series classical methods.

## Datasets (Downloadable 2D/Panel)
- Kaggle competitions (require Kaggle account/CLI)
  - M5 Forecasting – Walmart: Multi-item/store daily sales with prices/calendar. Metric: WRMSSE.
  - Corporación Favorita Grocery Sales: Item–store daily sales with promos/holidays.
  - Rossmann Store Sales: Store-level daily sales and promos.
  - Store Item Demand Forecasting: Per-store/product sales panel.
- Open archives / widely used academic sets
  - ETT (ETTh1/ETTh2/ETTm1/ETTm2): Multivariate energy/temperature; standard in Transformer papers.
  - Electricity Load Diagrams (UCI/Monash): 321 clients’ hourly electricity usage; multivariate panel.
  - Traffic (METR-LA, PEMS-BAY): Multivariate spatio-temporal road sensors.
  - Exchange/Weather (Monash TSF): Often used in multivariate benchmarks (some are univariate collections but can be stacked).

Notes
- Kaggle: use `kaggle competitions download` + acceptance of terms.
- ETT: available via GitHub/HuggingFace (programmatic access is easy).
- UCI/Monash: downloadable archives; many repos offer loaders.

## Benchmarks (ARIMA/Prophet context)
- Kaggle leaderboards (M5/Favorita/Rossmann): Top solutions are global tree ensembles (LightGBM/XGBoost/CatBoost) with hierarchical reconciliation and rich features. ARIMA/Prophet baselines exist in notebooks but are far from SOTA on large panels.
- ETT/Electricity/Exchange/Weather: Deep models (Informer, Autoformer, FEDformer, Pyraformer, TimesNet, PatchTST, iTransformer, TFT) dominate longer horizons and many-variable settings. Classical ARIMA/Prophet baselines typically underperform without heavy tuning.
- Practitioner libraries
  - StatsForecast (Nixtla): Production-grade AutoARIMA/ETS/Prophet with scalable backtesting.
  - Darts: Unified baselines (ARIMA/Prophet/ES) and deep models for quick comparisons.

Takeaway: For large multivariate panels, expect global ML/deep models to outperform per-series ARIMA/Prophet, especially on longer horizons.

## Modeling Approaches (Repo Adaptation)
- Global multivariate model
  - Input shape: `x ∈ R^{batch, seq_len, n_features}` (e.g., products/stores as channels).
  - Output shape: `ŷ ∈ R^{batch, pred_len, n_targets}` (often `n_targets == n_features`).
  - Current repo changes: 
    - Dataset: modify `TimeSeriesDataset` to return multivariate windows (stack items as features).
    - Models: set `input_dim = n_features`; projection `Linear(input_dim → d_model)` mixes variables; attention learns cross-item relations.
    - Heads: output `Linear(d_model → pred_len * n_targets)` then reshape to `(batch, pred_len, n_targets)`.
- Variable-aware enhancements (optional)
  - Series embeddings (ID/category) concatenated to inputs.
  - Variable selection/ gating (as in TFT).
  - Graph layers (GAT/GraphWaveNet) if a product relation graph is available.
- Causality
  - Encoder-only models remain causal if inputs contain only past observations.
  - Decoder-only must use masked self-attention (no unmasked cross-attn to “memory”). We should refactor our decoder-only block accordingly (see FINDINGS.md).

## Evaluation Protocols
- Static temporal split (baseline): 80/20 chronological or fixed last-H horizon; report MAE/RMSE/MAPE; for M5-style include WRMSSE.
- Moving-window backtesting: Rolling-origin 1-step or multi-step evaluation across windows; average metrics for robust comparison.
- Per-series vs global metrics: Report overall averages and distributions; consider scale-free metrics (MASE, sMAPE) when series scales differ.

## Concrete Repo Changes (Minimal Viable Pivot)
1) Data loading
- Add a multivariate loader that stacks selected series into a single panel matrix per dataset: `values[t] ∈ R^{n_features}`.
- Update `TimeSeriesDataset` to return `(x: [seq_len, n_features], y: [pred_len, n_targets])`.
- Scaling: start with global StandardScaler per-feature; optionally per-series scaling with learned offsets.

2) Models
- Transformer (encoder-only):
  - `input_dim = n_features`; project to `d_model`.
  - Use last hidden state → `Linear(d_model → pred_len * n_targets)` → reshape.
- Decoder-only:
  - Replace `TransformerDecoder` with masked self-attention only (no cross-attn to unmasked memory).
  - Train next-step (teacher forcing), roll out autoregressively.
- LSTM baseline:
  - Accept `[batch, seq_len, n_features]` input; head predicts `[pred_len, n_targets]`.

3) Evaluation
- Implement multivariate metrics runner to compute MAE/RMSE per target and aggregate.
- Support both static split and moving-window.
- For Kaggle-style (M5), add WRMSSE if we adopt that dataset.

4) Documentation
- Update README: add multivariate usage, datasets, and scripts.
- Add reproducible configs (seq_len, pred_len, feature selection) per dataset.

## Quick Start Plan (7–10 days)
- Day 1–2: Implement multivariate dataset class; add synthetic multivariate panel generator for testing.
- Day 2–3: Adapt Transformer/LSTM to multivariate I/O; refactor decoder-only to true masked self-attn.
- Day 3–4: Add multivariate evaluators (static + moving-window); baseline scripts; save results.
- Day 5–7: Run on ETT/Electricity; collect baseline ARIMA/Prophet (per-series) vs global Transformer/LSTM.
- Day 8–10: Optional: add simple variable embeddings; prepare figures and summary.

## Open Questions
- Targets: Forecast all channels vs a subset? (Default: all.)
- Scaling: Per-feature global vs per-series local scaling? (Start global.)
- Exogenous features: Calendar/promotions/prices (Kaggle) — do we integrate now or phase 2?
- Benchmarks: Which datasets to standardize on (ETT/Electricity first; M5 later due to complexity)?

## References (to verify/collect)
- Kaggle competitions: M5, Favorita, Rossmann, Store Item Demand.
- ETT and benchmark repos: Informer, Autoformer, FEDformer, TimesNet, PatchTST, iTransformer.
- Practitioner libs: Nixtla StatsForecast, Darts.

---
This plan isolates multivariate work without disrupting current univariate experiments and keeps changes minimal, focusing first on a working global Transformer/LSTM baseline with clear evaluation.

## Benchmark Sources (Links)

Datasets (download)
- M5 Forecasting (Walmart, WRMSSE): https://www.kaggle.com/competitions/m5-forecasting-accuracy
- Corporación Favorita Grocery Sales: https://www.kaggle.com/competitions/favorita-grocery-sales-forecasting
- Rossmann Store Sales: https://www.kaggle.com/competitions/rossmann-store-sales
- Store Item Demand Forecasting: https://www.kaggle.com/competitions/demand-forecasting-kernels-only
- ETT dataset (ETTh1/ETTm1 etc.): https://github.com/zhouhaoyi/ETDataset
- Electricity Load Diagrams (UCI): https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20112014
- Monash Time Series Forecasting Archive: https://forecastingdata.org/
- Traffic datasets (METR-LA/PEMS-BAY via repos):
  - DCRNN data: https://github.com/liyaguang/DCRNN/tree/master/data
  - Graph WaveNet data: https://github.com/zhiyongc/Graph-WaveNet/tree/master/data

Benchmarking papers/repos (multivariate baselines and tables)
- Informer: https://github.com/zhouhaoyi/Informer2020
- Autoformer: https://github.com/thuml/Autoformer
- FEDformer: https://github.com/MAZiqing/FEDformer
- TimesNet: https://github.com/thuml/TimesNet
- PatchTST: https://github.com/yuqinie/PatchTST
- iTransformer: https://github.com/thuml/iTransformer
- Temporal Fusion Transformer (reference implementation):
  - Paper: https://arxiv.org/abs/1912.09363
  - Community code: https://github.com/philipperemy/Temporal-Fusion-Transformers

Practitioner baseline frameworks
- StatsForecast (AutoARIMA/ETS/Prophet):
  - Repo: https://github.com/Nixtla/statsforecast
  - Docs: https://nixtla.github.io/statsforecast/
- Darts (ARIMA/Prophet/ES + deep models):
  - Repo: https://github.com/unit8co/darts
  - Docs: https://unit8co.github.io/darts/

Notes
- Exact numeric results vary by preprocessing, horizon, and metrics; use the above as sources to extract consistent benchmark tables or to reproduce results with standard scripts.
