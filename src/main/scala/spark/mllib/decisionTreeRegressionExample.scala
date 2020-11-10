package spark.mllib

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.{VectorAssembler, VectorIndexer}
import org.apache.spark.ml.regression.DecisionTreeRegressor
import org.apache.spark.sql.SparkSession

object decisionTreeRegressionExample {

  def main(args: Array[String]) {

    val spark = SparkSession.builder()
      .master("yarn")
      .appName("decisionTreeRegressionExample")
      .getOrCreate()

    val rawData = spark.read.format("csv").option("header",true).load("hdfs://master:9000/sparkMLlib/hour.csv")
    val data1 = rawData.select(rawData("season").cast("Double"),
      rawData("yr").cast("Double"),
      rawData("mnth").cast("Double"),
      rawData("hr").cast("Double"),
      rawData("holiday").cast("Double"),
      rawData("weekday").cast("Double"),
      rawData("workingday").cast("Double"),
      rawData("weathersit").cast("Double"),
      rawData("temp").cast("Double"),
      rawData("atemp").cast("Double"),
      rawData("hum").cast("Double"),
      rawData("windspeed").cast("Double"),
      rawData("cnt").cast("Double").alias("label")
    )

    val featuresArray = Array("season","yr","mnth","hr","holiday","weekday","workingday","weathersit","temp","atemp","hum","windspeed")
    val assembler = new VectorAssembler().setInputCols(featuresArray).setOutputCol("spark/mllib/features")
    val featureIndexer = new VectorIndexer().setInputCol("spark/mllib/features").setOutputCol("indexedFeatures").setMaxCategories(24)
    val Array(trainingData,testData) = data1.randomSplit(Array(0.7,0.3),12)
    val dt = new DecisionTreeRegressor().setLabelCol("label").setFeaturesCol("indexedFeatures").setMaxBins(64).setMaxDepth(15)
    val pipeline = new Pipeline().setStages(Array(assembler,featureIndexer,dt))
    val model = pipeline.fit(trainingData)
    val predictions = model.transform(testData)

    val evaluator = new RegressionEvaluator().setLabelCol("label").setPredictionCol("prediction").setMetricName("rmse")
    val rmse = evaluator.evaluate(predictions)
    println("rmse: " + rmse)
  }
}
