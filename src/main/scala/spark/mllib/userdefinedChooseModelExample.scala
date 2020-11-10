//package spark.mllib
//
//
//import org.apache.spark.ml.recommendation.{ALS, ALSModel}
//import org.apache.spark.sql.{DataFrame, SparkSession}
//
//object userdefinedChooseModelExample {
//  def main(args: Array[String]): Unit = {
//    val spark = SparkSession
//      .builder()
//      .master("yarn")
//      .appName("userdefinedChooseModelExample")
//      .config("spark.testing.memory", "471859200")
//      .getOrCreate()
//    val ratings = spark.read
//      .textFile("hdfs://master:9000/sparkMLlib/sample_movielens_ratings.txt")
//      .map(parseRating)
//      .toDF()
//
//
//    val splits = ratings.randomSplit(Array(0.6, 0.2, 0.2), 12)
//
//
//    val training = splits(0).cache()
//    val validation = splits(1).toDF().cache()
//    val test = splits(2).toDF().cache()
//
//    val numTraining = training.count()
//    val numValidation = validation.count()
//    val numTest = test.count()
//
//    println("numTraining: " + numTraining + " " + "numValidation: " + numValidation + " " + "numTest: " + numTest)
//
//
//    val ranks = List(20, 40)
//    val lambdas = List(0.01, 0.1)
//    val numIters = List(5, 10)
//    var bestModel: Option[ALSModel] = None
//    var bestValidaionRmse = Double.MaxValue
//    var bestRank = 0
//    var bestLambda = 1.0
//    var bestNumIter = 1
//
//    for (rank <- ranks; lambda <- lambdas; numIter <- numIters) {
//
//      val als = new ALS()
//        .setMaxIter(numIter)
//        .setRegParam(lambda)
//        .setRank(rank)
//        .setUserCol("userId")
//        .setItemCol("movieId")
//        .setRatingCol("rating")
//
//      val model = als.fit(training)
//
//      val validationRmse = computeRmse(model, validation, numValidation)
//
//      println ("RMSE(validation) = " + validationRmse + " for the model trained with rank = " +
//      rank + ", lambda = " + lambda + ", and numIter = " + numIter + ".")
//
//      if (validationRmse < bestValidaionRmse) {
//        bestModel = Some(model)
//        bestValidaionRmse = validationRmse
//        bestRank = rank
//        bestLambda = lambda
//        bestNumIter = numIter
//      }
//    }
//
//    val testRmse = computeRmse(bestModel.get,test,numTest)
//    println("testRmse: ",testRmse)
//
//    println("The best model was trained with rank = " + bestRank + " and lambda = " + bestLambda + ", and numIter = "
//    + bestNumIter + ", and its RMSE on the test set is " + testRmse + ".")
//  }
//
//  case class Rating(userId: Int, movieId: Int, rating: Float, timestamp: Long)
//  def parseRating(str: String): Rating = {
//    val fields = str.split("::")
//    assert(fields.size == 4)
//    Rating(fields(0).toInt,
//           fields(1).toInt,
//           fields(2).toFloat,
//           fields(3).toLong)
//  }
//
//  def computeRmse(model: ALSModel, data: DataFrame, n: Long): Double = {
//    val predictions = model.transform(data)
//    //val p1 = predictions.rdd.map { x => ((x(0),x(1)),x(2))}
//    //val p2 = predictions.rdd.map { x => ((x(0),x(1)),x(4))}
//    //val p3 = p1.join(p2).values
//
//    val p1 = predictions.rdd
//      .map { x =>
//        ((x(0), x(1)), x(2))
//      }.join(predictions.rdd.map { x =>
//        ((x(0), x(1)), x(4))
//      }).values
//    math.sqrt(p1
//      .map(x =>
//        (x._1.toString.toDouble - x._2.toString.toDouble) * (x._1.toString.toDouble - x._2.toString.toDouble))
//      .reduce(_ + _) / n)
//  }
//}
