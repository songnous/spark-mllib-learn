package spark.mllib.recommendedSystem

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.recommendation.ALS
import org.apache.spark.sql.{DataFrame, Dataset, SparkSession}
import spark.mllib.recommendedSystem.dataDispose.Rating

object trainModel {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder()
      .master("local")
      .appName("trainModel")
      .config("spark.testing.memory", "471859200")
      .getOrCreate()

    val path = "hdfs://master:9000/sparkMLlib/u.data"
    val maxIter = 10
    val rank = 10
    val regParam = 0.01
    val userCol = "userId"
    val itemCol = "movieId"
    val ratingCol = "rating"
    val trainPercentag = 0.8

    getPredictions(spark,path,maxIter,rank,regParam,userCol,itemCol,ratingCol,trainPercentag)
  }
  def splitSampleToTrainAndTest(spark: SparkSession,path: String,trainPercentag: Double):(Dataset[Rating],Dataset[Rating]) = {
    val rating = dataDispose.fromHdfsToLoadData(spark,path)
    val Array(training,test) = rating.randomSplit(Array(trainPercentag,1-trainPercentag))
    (training,test)
  }
  def trainCfMode(spark: SparkSession,path: String,maxIter: Int,rank: Int,regParam: Double,
                  userCol: String,itemCol: String,ratingCol: String,trainPercentag :Double) :DataFrame = {

    val als = new ALS()
      .setMaxIter(maxIter)
      .setRank(rank)
      .setRegParam(regParam)
      .setNonnegative(true)
      .setUserCol(userCol)
      .setItemCol(itemCol)
      .setRatingCol(ratingCol)
    val pipeline = new Pipeline().setStages(Array(als))

    val (training,test) = splitSampleToTrainAndTest(spark,path,trainPercentag)
    val model = pipeline.fit(training)
    model.transform(test)
  }
  def getPredictions(spark: SparkSession,path: String,maxIter: Int,rank: Int,regParam: Double,
                     userCol: String,itemCol: String,ratingCol: String,trainPercentag :Double) = {
    val predictions = trainCfMode(spark,path,maxIter,rank,regParam,userCol,itemCol,ratingCol,trainPercentag)

    predictions.show()
  }

}
