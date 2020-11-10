package spark.mllib.modelChooseOptimization

import org.apache.spark.ml.recommendation.{ALS, ALSModel}
import org.apache.spark.sql.{DataFrame, SparkSession}

object userdefinedChooseModelExample {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder()
      .master("local")
      .appName("userdefinedChooseModelExample")
      .config("spark.testing.memory", "4294967296")
      .getOrCreate()

    import spark.implicits._
    val ratings = spark.read
      .textFile("hdfs://master:9000/sparkMLlib/sample_movielens_ratings.txt")
      .map(parseRating)
      .toDF()

    // 将样本评分表分成 3 个部分，分别用于训练(60%)，校验(20%)，和测试(20%)

    val splits = ratings.randomSplit(Array(0.6, 0.2, 0.2), 12)

//    // 把训练样本缓存起来，加快运算速度

    val training = splits(0).cache()
    val validation = splits(1).toDF().cache()
    val test = splits(2).toDF().cache()

    //  计算各集合总数
    val numTraining = training.count()
    val numValidation = validation.count()
    val numTest = validation.count()

    println(
      "numTraining: " + numTraining + " " + "numValidation: " + numValidation + " " + "numTest: " + numTest)

    // 训练不同参数下的模型，并在校验集中验证，获取最佳参数下的模型

    val ranks = List(20, 40)
    val lambdas = List(0.01, 0.1)
    val numIters = List(5, 10)
    var bestModel: Option[ALSModel] = None
    var bestValidaionRmse = Double.MaxValue
    var bestRank = 0
    var bestLambda = 1.0
    var bestNmuIter = 1

    for (rank <- ranks; lambda <- lambdas; numIter <- numIters) {

      val als = new ALS()
        .setMaxIter(numIter)
        .setRegParam(lambda)
        .setRank(rank)
        .setUserCol("userId")
        .setItemCol("movieId")
        .setRatingCol("rating")

      val model = als.fit(training)

          val validationRmse = computeRmse(model, validation, numValidation)

      println(
        "RMSE(validation) = " + validationRmse + " for the model trained with rank = " +
          rank + ", lambda = " + lambda + ", and numIter = " + numIter + ".")

      if (validationRmse < bestValidaionRmse) {
        bestModel = Some(model)
        bestValidaionRmse = validationRmse
        bestRank = rank
        bestLambda = lambda
        bestNmuIter = numIter
      }
    }
  }

  case class Rating(userId: Int, movieId: Int, rating: Float, timestamp: Long)
  def parseRating(str: String): Rating = {
    val fields = str.split("::")
    assert(fields.size == 4)
    Rating(fields(0).toInt,
           fields(1).toInt,
           fields(2).toFloat,
           fields(3).toLong)
  }

  def computeRmse(model: ALSModel, data: DataFrame, n: Long): Double = {
    val predictions = model.transform(data)
    val p1 = predictions.rdd.map { x => ((x(0),x(1)),x(2))}
    val p2 = predictions.rdd.map { x => ((x(0),x(1)),x(4))}
    val p3 = p1.join(p2).values

//    val p1 = predictions.rdd
//      .map { x =>
//        ((x(0), x(1)), x(2))
//      }.join(predictions.rdd.map { x =>
//        ((x(0), x(1)), x(4))
//      })
//      .values
    math.sqrt(p3
      .map(x =>
        (x._1.toString.toDouble - x._2.toString.toDouble) * (x._1.toString.toDouble - x._2.toString.toDouble))
      .reduce(_ + _) / n)
  }
}
