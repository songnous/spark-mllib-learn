package spark.mllib.recommendedSystem

import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.sql.SparkSession

object evaluationModel {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder()
      .master("local")
      .appName("evaluationModel")
      .config("spark.testing.memory", "471859200")
      .getOrCreate()

    val input = "hdfs://master:9000/sparkMLlib/u.data"
    val maxIter = 10
    val rank = 10
    val RegParam = 0.01
    val userCol = "userId"
    val itemCol = "movieId"
    val ratingCol = "rating"
    val trainPercentag = 0.8

    val (rmse,rmse1) = evaluationModelRMSE(spark,input,maxIter,rank,RegParam,userCol,itemCol,ratingCol,trainPercentag)

    println("rmse: " + rmse)
    println("rmse1: " + rmse1)

    val (predictionNanCount,predictionsCount,predictions1Count) =
      getPredictions(spark,input,maxIter,rank,RegParam,userCol,itemCol,ratingCol,trainPercentag)

    println("predictionsCount: " + predictionsCount)
    println("predictions1Count: " + predictions1Count)
    println("predictionNanCount " + predictionNanCount)
  }


  def evaluationModelRMSE(spark: SparkSession,path: String,maxIter: Int,rank: Int,regParam: Double, userCol: String,
                          itemCol: String,ratingCol: String,trainPercentag :Double): Tuple2[Double,Double]= {
    val predictions = trainModel.trainCfMode(spark,path,maxIter,rank,regParam,userCol,itemCol,ratingCol,trainPercentag)

    // val predictionNanCount = predictions.filter(predictions("prediction").isNaN).select("userId","movieId","rating","prediction").count()
    val predictions1 = predictions.na.drop()

    val evaluator = new RegressionEvaluator()
      .setMetricName("rmse")
      .setLabelCol("rating")
      .setPredictionCol("prediction")
    val rmse = evaluator.evaluate(predictions)
    val rmse1 = evaluator.evaluate(predictions1)
    (rmse,rmse1)
  }

  def getPredictions(spark: SparkSession,path: String,maxIter: Int,rank: Int,regParam: Double,
                     userCol: String,itemCol: String,ratingCol: String,trainPercentag :Double) :(Long,Long,Long) = {
    val predictions = trainModel.trainCfMode(spark,path,maxIter,rank,regParam,userCol,itemCol,ratingCol,trainPercentag)
    val predictionNanCount = predictions.filter(predictions("prediction").isNaN)
      .select("userId","movieId","rating","prediction").count()

    val  predictions1 = predictions.na.drop()

    println("predictions.show(5): ")
    predictions.show(5)
    println("predictions1.show(5): ")
    predictions1.show(5)
    (predictionNanCount,predictions.count(),predictions1.count())
  }
}
