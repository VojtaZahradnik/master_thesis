# Evaluation

Whole evaluation we can find in [evl.py](../../research_task/src/modules/evl.py). We need to have evaluation module to compare many models based on many metrics.

## Metrics

I implemented many metrics, also known from statistics:

- RMSE: Root mean square error(mostly used)
- R^2: Not great metric, because we need to compare two models on out-of-sample prediction.
- Standard deviation: Very basic metric.
- MSE: Mean square error.
- MAE: Mean average error.
- Residual metric: Sum of residuals.
- Graph: Visualize prediction could be sometimes the best solution.