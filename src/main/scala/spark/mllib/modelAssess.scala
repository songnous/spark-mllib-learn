package spark.mllib

import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.{BinaryClassificationEvaluator, MulticlassClassificationEvaluator}
import org.apache.spark.mllib.evaluation.RegressionMetrics
import org.apache.spark.sql.{DataFrame, SparkSession}

object modelAssess {
  def main(args: Array[String]): Unit = {
    val path = args(0)
    val spark: SparkSession = SparkSession.builder
      .appName("modelAsess")
      .master("yarn")
      .config("spark.testing.memory", "471859200")
      .getOrCreate
    val data = spark.read.format("libsvm").load(path)
    val Array(trainingData, testData) =
      data.randomSplit(Array(0.7, 0.3), seed = 1234L)
    val lr = new LogisticRegression()
      .setThreshold(0.6)
      .setMaxIter(10)
      .setRegParam(0.3)
      .setElasticNetParam(0.8)
    val lrMode = lr.fit(trainingData)
    val predictions = lrMode.transform(testData)
    predictions.show()

    val evaluator = new BinaryClassificationEvaluator()
      .setLabelCol("label")
    val accuracy = evaluator.evaluate(predictions)
    val rm2 = new RegressionMetrics(
      predictions
        .select("prediction", "label")
        .rdd
        .map(x => (x(0).asInstanceOf[Double], x(1).asInstanceOf[Double])))

    println("MSE: ", rm2.meanSquaredError)
    println("MAE: ", rm2.meanAbsoluteError)
    println("RMSE Squared: ", rm2.rootMeanSquaredError)

    val binaryClassificationEvaluator = new BinaryClassificationEvaluator()
    val multiclassClassificationEvaluator: MulticlassClassificationEvaluator =
      new MulticlassClassificationEvaluator()

    printlnMetricMulti("f1", predictions, multiclassClassificationEvaluator)
    printlnMetricMulti("weightedPrecision",
                       predictions,
                       multiclassClassificationEvaluator)
    printlnMetricMulti("weightedRecall",
                       predictions,
                       multiclassClassificationEvaluator)
    printlnMetricMulti("accuracy",
                       predictions,
                       multiclassClassificationEvaluator)

    printlnMetricbinary("areaUnderROC",
                        binaryClassificationEvaluator,
                        predictions)
    printlnMetricbinary("areaUnderPR",
                        binaryClassificationEvaluator,
                        predictions)


    // A error of "value $ is not StringContext member" is reported if you don't add following line
/**
    println(
      predictions
        .filter($"label" === $"prediction")
        .filter($"label" === 1)
        .count)

    println(
      predictions
        .filter($"label" === $"prediction")
        .filter($"prediction" === 0)
        .count)

    println(
      predictions
        .filter($"label" !== $"prediction")
        .filter($"prediction" === 0)
        .count)

    println(
      predictions
        .filter($"label" !== $"prediction")
        .filter($"prediction" === 1)
        .count)
 */
  }

  def printlnMetricMulti(
      metricsName: String,
      predictions: DataFrame,
      multiclassClassificationEvaluatdor: MulticlassClassificationEvaluator)
    : Unit = {
    //val multiclassClassificationEvaluatdor = new MulticlassClassificationEvaluator()
    println(
      metricsName + " = " + multiclassClassificationEvaluatdor
        .setMetricName(metricsName)
        .evaluate(predictions))
  }

  def printlnMetricbinary(
      metricsName: String,
      binaryClassificationEvaluator: BinaryClassificationEvaluator,
      predictions: DataFrame): Unit = {

    println(
      metricsName + " = " + binaryClassificationEvaluator
        .setMetricName(metricsName)
        .evaluate(predictions))
  }
}
