package spark.mllib

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.{OneHotEncoder, SQLTransformer, VectorAssembler}
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.sql.SparkSession

object regressionModelExample {

  def main(args: Array[String]) {

    val spark = SparkSession.builder()
      .master("yarn")
      .appName("regressionModelExample")
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

    val data2 = new OneHotEncoder().setInputCol("season").setOutputCol("seasonVec")
    val data3 = new OneHotEncoder().setInputCol("yr").setOutputCol("yrVec")
    val data4 = new OneHotEncoder().setInputCol("mnth").setOutputCol("mnthVec")
    val data5 = new OneHotEncoder().setInputCol("hr").setOutputCol("hrVec")
    val data6 = new OneHotEncoder().setInputCol("holiday").setOutputCol("holidayVec")
    val data7 = new OneHotEncoder().setInputCol("weekday").setOutputCol("weekdayVec")
    val data8 = new OneHotEncoder().setInputCol("workingday").setOutputCol("workingdayVec")
    val data9 = new OneHotEncoder().setInputCol("weathersit").setOutputCol("weathersitVec")

    val pipeline_en = new Pipeline().setStages(Array(data2,data3,data4,data5,data6,data7,data8,data9))
    val data = pipeline_en.fit(data1).transform(data1)

    // val assembler = new VectorAssembler().setInputCols(Array("seasonVec","yrVec","hrVec","holidayVec","weekdayVec","workingdayVec","temp","atemp","hum","windspeed")).setOutputCol("features")
    val assembler = new VectorAssembler().setInputCols(Array("seasonVec","yrVec","hrVec","holidayVec","weekdayVec","workingdayVec","temp","hum","windspeed")).setOutputCol("spark/mllib/features")

    val sqlTrans = new SQLTransformer().setStatement("SELECT *, SQRT(label) AS label1 FROM __THIS__")
    val Array(trainingData,testData) = data.randomSplit(Array(0.7,0.3),12)
    //val lr = new LinearRegression()
    //  .setLabelCol("label")
    //  .setFeaturesCol("features")
    //  .setFitIntercept(true)
    //  .setMaxIter(20)
    //  .setRegParam(0.3)
    //  .setElasticNetParam(0.8)

    val lr = new LinearRegression()
      .setLabelCol("label1")
      .setFeaturesCol("spark/mllib/features")
      .setFitIntercept(true)

    //val pipeline = new Pipeline().setStages(Array(assembler,lr))
    val pipeline = new Pipeline().setStages(Array(assembler,sqlTrans,lr))

    val paramGrid = new ParamGridBuilder()
      .addGrid(lr.elasticNetParam,Array(0.0,0.8,1.0))
      .addGrid(lr.regParam,Array(0.1,0.3,0.5))
      .addGrid(lr.maxIter,Array(15,20,30))
      .build()

    val evaluator = new RegressionEvaluator().setLabelCol("label1").setPredictionCol("prediction").setMetricName("rmse")

    val trainValidationSplit = new TrainValidationSplit()
      .setEstimator(pipeline)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid)
      .setTrainRatio(0.8)

    val model = trainValidationSplit.fit(trainingData)

    println("model.getEstimatorParamMaps: ")
    model.getEstimatorParamMaps.foreach(println(_))

    println("model.getEvaluator.extractParamMap: ")
    println(model.getEvaluator.extractParamMap())

    println("model.getEvaluator.isLargerBetter: ")
    println(model.getEvaluator.isLargerBetter)

    val predictions = model.transform(testData)

    //val evaluator = new RegressionEvaluator().setLabelCol("label").setPredictionCol("prediction").setMetricName("rmse")
    val rmse = evaluator.evaluate(predictions)
    println("rmse: " + rmse)

    predictions.select("spark/mllib/features","label","label1","prediction").show(false)
  }
}
