# ✈️ Flight Price Prediction (PySpark on Google Cloud)

End-to-end ML pipeline using **PySpark on Google Cloud Dataproc Serverless** to predict airline ticket prices.  
Dataset: [Kaggle – Flight Price Prediction](https://www.kaggle.com/datasets/shubhambathwal/flight-price-prediction).

## What this does
- Loads clean dataset from **Google Cloud Storage**
- Feature engineering:
  - `days_left` (days before departure)
  - `stops` → integer (`stops_int`)
  - Categorical features: `airline, source_city, destination_city, departure_time, arrival_time, class`
- Trains **Gradient-Boosted Trees** (PySpark MLlib Pipeline)
- Saves **model** and **metrics** to GCS

## Proven performance (artifact)
See **[results/metrics.json](results/metrics.json)** (direct output from latest Dataproc run).

## No look-ahead
- Splits are done **before** fitting; encoders learn only from **train**.
- Added **deduplication** to avoid twin rows across train/test.

## Reproduce
```bash
gcloud dataproc batches submit pyspark \
  --region=us-central1 \
  --deps-bucket=gs://YOUR_BUCKET \
  train_spark.py -- \
  gs://YOUR_BUCKET/data/Clean_Dataset.csv \
  gs://YOUR_BUCKET/artifacts/spark_model \
  gs://YOUR_BUCKET/artifacts/metrics'
