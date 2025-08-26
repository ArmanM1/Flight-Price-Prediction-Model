
import sys, json, re
from pyspark.sql import SparkSession, functions as F, types as T
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.regression import GBTRegressor
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator

# Args: 1) input CSV path, 2) model output path, 3) metrics output path
inp, model_out, metrics_out = sys.argv[1], sys.argv[2], sys.argv[3]

spark = SparkSession.builder.appName("flight-fares-train-clean").getOrCreate()
df = spark.read.option("header", True).csv(inp)

# Drop junk index if present
if "Unnamed: 0" in df.columns:
    df = df.drop("Unnamed: 0")

# Ensure numeric dtypes
df = (df
    .withColumn("duration",  F.col("duration").cast("double"))
    .withColumn("days_left", F.col("days_left").cast("int"))
    .withColumn("price",     F.col("price").cast("double"))
)

# Map 'stops' text -> integer
def stops_to_int(s):
    if s is None: return None
    s = str(s).strip().lower()
    if s in {"non-stop","non stop","zero","0"}: return 0
    if s in {"one","1","1 stop","1-stop"}:      return 1
    if s in {"two","2","2 stop","2-stop"}:      return 2
    if s in {"three","3"}:                      return 3
    return None

stops_udf = F.udf(stops_to_int, T.IntegerType())
df = df.withColumn("stops_int", stops_udf(F.col("stops")))

# Keep only learnable rows
df = df.dropna(subset=["price","duration","days_left","stops_int"])

# NEW: Deduplicate to avoid leakage via twins across splits
dedup_keys = ["airline","source_city","destination_city","departure_time",
              "arrival_time","class","duration","days_left","stops_int","price"]
df = df.dropDuplicates(dedup_keys)

# Categorical + numeric features
cat_cols = [c for c in ["airline","source_city","destination_city",
                        "departure_time","arrival_time","class"] if c in df.columns]
num_cols = ["duration","days_left","stops_int"]

# Encode categoricals and assemble feature vector
indexers = [StringIndexer(handleInvalid="keep", inputCol=c, outputCol=f"{c}_idx") for c in cat_cols]
enc = OneHotEncoder(inputCols=[f"{c}_idx" for c in cat_cols],
                    outputCols=[f"{c}_oh"  for c in cat_cols],
                    handleInvalid="keep")
assembler = VectorAssembler(inputCols=[f"{c}_oh" for c in cat_cols] + num_cols,
                            outputCol="features")

# Model
gbt = GBTRegressor(featuresCol="features", labelCol="price",
                   maxDepth=6, maxIter=60, stepSize=0.05, seed=42)

pipeline = Pipeline(stages=indexers + [enc, assembler, gbt])

# Train / test split
train, test = df.randomSplit([0.8, 0.2], seed=42)
model = pipeline.fit(train)
pred  = model.transform(test)

# Evaluate
evalr = RegressionEvaluator(labelCol="price", predictionCol="prediction")
rmse = evalr.evaluate(pred, {evalr.metricName: "rmse"})
mae  = evalr.evaluate(pred, {evalr.metricName: "mae"})
r2   = evalr.evaluate(pred, {evalr.metricName: "r2"})

# Save artifacts
model.write().overwrite().save(model_out)
spark.createDataFrame([{"rmse": rmse, "mae": mae, "r2": r2}]) \
     .coalesce(1).write.mode("overwrite").json(metrics_out)

print(json.dumps({"rmse": rmse, "mae": mae, "r2": r2}, indent=2))

