package spark.mllib.recommendedSystem

import org.apache.spark.ml.recommendation.{ALS, ALSModel}
import org.apache.spark.sql.{DataFrame, SparkSession}

object ALSModelOptimize {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder()
      .master("local")
      .appName("ALSModelOptimize")
      .config("spark.testing.memory", "471859200")
      .getOrCreate()


    val input = "hdfs://master:9000/sparkMLlib/u.data"
    val ratings = dataDispose.fromHdfsToLoadData(spark,input)
    val splits = ratings.randomSplit(Array(0.6,0.2,0.2),12)

    val training = splits(0).cache()
    val validation = splits(1).toDF.cache()
    val test = splits(2).toDF.cache()

    val numTraning = training.count()
    val numValidation = validation.count()
    val numTest = test.count()

    println("numTraning: " + numTraning)
    println("numValidation: " + numValidation)
    println("numTest: " + numTest)

    val ranks = List(10,20,30)
    val lambdas = List(0.01,0.1,0.4)
    val numIters = List(5,10,20)
    var bestModel: Option[ALSModel] = None
    var bestValidationRmse = Double.MaxValue
    var bestRank = 0
    var bestLambda = 1.0
    var bestNumIter = 1

    for (rank <- ranks;lambda <- lambdas;numIter <- numIters) {
      val als = new ALS()
        .setMaxIter(numIter)
        .setRegParam(lambda)
        .setRank(rank)
        .setNonnegative(true)
        .setUserCol("userId")
        .setItemCol("movieId")
        .setRatingCol("rating")

      val model = als.fit(training)
      val validationRmse = computeRmse(model,validation,numValidation)
      println("RMSE(validation) = " + validationRmse + " for the model trained with rank = " + rank +
      ",lambda =" + lambda + ", and numIter = " + numIter + ".")

      if(validationRmse < bestValidationRmse) {
        bestModel = Some(model)
        bestValidationRmse = validationRmse
        bestRank = rank
        bestLambda = lambda
        bestNumIter = numIter
      }
    }
  }
  def computeRmse(model :ALSModel,data :DataFrame,n: Long) : Double = {
    val predictions = model.transform(data)
    val p1 = predictions.na.drop().rdd.map{ x =>
      ((x(0),x(1)),x(2))
    }.join(predictions.na.drop().rdd.map { x =>
      ((x(0),x(1)),x(4))
    } ).values
    math.sqrt(p1.map( x =>
      (x._1.toString.toDouble - x._2.toString.toDouble) * (x._1.toString.toDouble - x._2.toString.toDouble)).reduce(_+_)/n)
  }
}
