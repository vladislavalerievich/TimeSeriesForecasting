# Multivariate Time Series Forecasting Using Linear RNNs

Run

```bash
python -m src.training.trainer
python -m  src.synthetic_generation.plot_mts
```

If you encounter `ModuleNotFoundError: No module named 'src'` then add the project root (`~/TimeSeriesForecasting`) to `PYTHONPATH`. Run:

```bash
export PYTHONPATH=.
```

Create additional functions to precompute the necessary statistics to for the StaticFeaturesDataContainer.  Or would it be better to have these as a separate functions, and not inside StaticFeaturesDataContainer? Before starting to write the implementation, ask me the necessary clarifying questions and present me with design choices
 