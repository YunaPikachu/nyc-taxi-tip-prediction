## This section demonstrates how to use a PySpark ML pipeline for this project:

### Data preparation → feature engineering → pipeline construction → cross-validated training → evaluation → saving outputs.

## Data Preparation

Input Data: gs://my-bigdata-project-yuna/cleaned/ (Parquet format).

Filter: keep only rows where payment_type == 1 (credit card trips).

  → Credit card transactions are more reliable for tip records.

Rare Location Filtering: I remove PULocationIDs with fewer than 500 trips.

  → This reduces noise, prevents overly sparse one-hot encodings, and improves both accuracy and training speed.


## Feature Engineering (inside the Pipeline)

#### Time-based Features

Pickup Hour (pickup_hour) from tpep_pickup_datetime.

Day of Week (pickup_dayofweek) from tpep_pickup_datetime.

→ Capture rush-hour and weekday vs. weekend tipping patterns.

#### Categorical Features

PULocationID, DOLocationID, pickup_hour, pickup_dayofweek

Processed in the pipeline using:

StringIndexer → converts categories into numeric indices.

OneHotEncoder → creates binary vectors for machine learning.

#### Numerical Features

passenger_count, trip_distance, fare_amount

Scaled inside the pipeline with StandardScaler for normalization.

#### Feature Assembly

All categorical and numeric features are combined with VectorAssembler into one features column.

This assembled vector is passed directly into the regression model.

## Pipeline & Model Training
### Pipeline Design

The PySpark Pipeline integrates all steps into one object:

StringIndexer → OneHotEncoder → VectorAssembler (numeric) → StandardScaler → VectorAssembler (all features) → LinearRegression

#### Model Training

Linear Regression (baseline regression model).

Optimized using CrossValidator with 2-fold cross-validation.

Hyperparameter tuned on regParam = 0.01.

#### Model Evaluation

Metric: Root Mean Squared Error (RMSE).

On the 1% sample, RMSE ≈ 2.27.

On the full dataset, RMSE ≈ 2.35.

Interpretation: the model’s tip predictions are off by about $2.30 on average per trip.






```
from pyspark.sql import SparkSession
from pyspark.sql.functions import hour, dayofweek, col, count
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
from pyspark.ml.regression import LinearRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

# -------------------------------
# STEP 1: Testing pipeline on 1% sample
# -------------------------------
spark = SparkSession.builder.appName("TaxiTipPrediction").getOrCreate()
print("=== STEP 1: Testing pipeline on 1% sample ===")

# Load data and filter for "payment_type"
df_sample = spark.read.parquet("gs://my-bigdata-project-yuna/cleaned/")
df_sample = df_sample.filter(col("payment_type") == 1)

# Take 1% sample
df_sample = df_sample.sample(False, 0.001, seed=42).persist()

# Extract time-based features
df_sample = df_sample.withColumn("pickup_hour", hour("tpep_pickup_datetime"))
df_sample = df_sample.withColumn("pickup_dayofweek", dayofweek("tpep_pickup_datetime"))

# Define features
categorical_cols = ["PULocationID", "DOLocationID", "pickup_hour", "pickup_dayofweek"]
numeric_cols = ["passenger_count", "trip_distance", "fare_amount"]
target = "tip_amount"

# Index and encode categorical features
indexers = [StringIndexer(inputCol=c, outputCol=c+"_idx", handleInvalid="keep") for c in categorical_cols]
encoders = [OneHotEncoder(inputCol=c+"_idx", outputCol=c+"_vec") for c in categorical_cols]

# Scale numeric features
assembler_numeric = VectorAssembler(inputCols=numeric_cols, outputCol="numeric_vec")
scaler = StandardScaler(inputCol="numeric_vec", outputCol="scaled_numeric", withMean=True, withStd=True)

# Assemble all features
all_features = [c+"_vec" for c in categorical_cols] + ["scaled_numeric"]
assembler = VectorAssembler(inputCols=all_features, outputCol="features")

# Initialize model
lr = LinearRegression(featuresCol="features", labelCol=target)

# Create pipeline
pipeline = Pipeline(stages=indexers + encoders + [assembler_numeric, scaler, assembler, lr])

# Split sample
train_sample, test_sample = df_sample.randomSplit([0.8, 0.2], seed=42)

# Set up evaluator
evaluator = RegressionEvaluator(labelCol=target, predictionCol="prediction", metricName="rmse")

# Setup cross-validation
paramGrid = ParamGridBuilder().addGrid(lr.regParam, [0.01]).build()
crossval = CrossValidator(estimator=pipeline,
                          estimatorParamMaps=paramGrid,
                          evaluator=evaluator,
                          numFolds=2,
                          parallelism=1)

# Fit and evaluate sample model
cv_model_sample = crossval.fit(train_sample)
pred_sample = cv_model_sample.transform(test_sample)
rmse_sample = evaluator.evaluate(pred_sample)
print(f"Sample RMSE: {rmse_sample:.2f}")

# -------------------------------
# STEP 2: Running pipeline on full dataset
# -------------------------------
print("=== STEP 2: Running pipeline on full dataset ===")
spark = SparkSession.builder.appName("TaxiTipPredictionFull").getOrCreate()

# Load full cleaned dataset
df_full = spark.read.parquet("gs://my-bigdata-project-yuna/cleaned/")
df_full = df_full.filter(col("payment_type") == 1)

# Extract time-based features
df_full = df_full.withColumn("pickup_hour", hour("tpep_pickup_datetime"))
df_full = df_full.withColumn("pickup_dayofweek", dayofweek("tpep_pickup_datetime"))

# Count location frequencies
location_counts = df_full.groupBy("PULocationID").agg(count("*").alias("count"))
top_locations = location_counts.filter(col("count") > 500).select("PULocationID").rdd.flatMap(lambda x: x).collect()

# Filter out rare PULocationID
df_full = df_full.filter(col("PULocationID").isin(top_locations))

# Define features
categorical_cols = ["PULocationID", "DOLocationID", "pickup_hour", "pickup_dayofweek"]
numeric_cols = ["passenger_count", "trip_distance", "fare_amount"]
target = "tip_amount"

# Index and encode categorical features
indexers = [StringIndexer(inputCol=c, outputCol=c+"_idx", handleInvalid="keep") for c in categorical_cols]
encoders = [OneHotEncoder(inputCol=c+"_idx", outputCol=c+"_vec") for c in categorical_cols]

# Scale numeric features
assembler_numeric = VectorAssembler(inputCols=numeric_cols, outputCol="numeric_vec")
scaler = StandardScaler(inputCol="numeric_vec", outputCol="scaled_numeric", withMean=True, withStd=True)

# Assemble all features
all_features = [c+"_vec" for c in categorical_cols] + ["scaled_numeric"]
assembler = VectorAssembler(inputCols=all_features, outputCol="features")

# Linear Regression
lr = LinearRegression(featuresCol="features", labelCol=target)

# Pipeline
pipeline = Pipeline(stages=indexers + encoders + [assembler_numeric, scaler, assembler, lr])

# Split full data
train_full, test_full = df_full.randomSplit([0.8, 0.2], seed=42)

# Evaluator
evaluator = RegressionEvaluator(labelCol=target, predictionCol="prediction", metricName="rmse")

# Grid search for tuning
paramGrid = ParamGridBuilder().addGrid(lr.regParam, [0.01]).build()

# CrossValidator
crossval = CrossValidator(estimator=pipeline,
                          estimatorParamMaps=paramGrid,
                          evaluator=evaluator,
                          numFolds=2,
                          parallelism=1)

# Fit and evaluate on full dataset
cv_model_full = crossval.fit(train_full)
pred_full = cv_model_full.transform(test_full)
rmse_full = evaluator.evaluate(pred_full)
print(f"RMSE on subsample: {rmse_full:.2f}")

# Apply feature pipeline to full data
feature_pipeline = Pipeline(stages=indexers + encoders + [assembler_numeric, scaler, assembler])
feature_model = feature_pipeline.fit(df_full)  # You already have df_full defined
df_with_features = feature_model.transform(df_full)

# Save to GCS
df_with_features.write.mode("overwrite").parquet("gs://my-bigdata-project-yuna/trusted/full_features/")
```
