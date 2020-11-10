package spark.mllib.recommendedSystem

import org.apache.spark.sql.{Dataset, SparkSession}
object dataDispose {
  def main(args: Array[String]): Unit = {

    val spark = SparkSession
          .builder()
          .master("local")
          .appName("loadData")
          .config("spark.testing.memory", "471859200")
          .getOrCreate()
    val path = "hdfs://master:9000//sparkMLlib/u.data"
    getDataDesc(spark,path)
//    val ratings = fromHdfsToLoadData(spark,path)
//    print(ratings.take(10))

  }

  case class Rating(userId: Int,movieId: Int,rating: Float,timestamp : Long)
  def parseRating(str: String) : Rating = {
    val fields = str.split("\t")
    //  如果 fields.size ！= 4  则程序终止，而且会报错
    assert(fields.size == 4)
    Rating(fields(0).toInt,fields(1).toInt,fields(2).toFloat,fields(3).toLong)
  }


  def fromHdfsToLoadData(spark: SparkSession,path: String):Dataset[Rating] = {

    import spark.implicits._
    (spark.read.textFile(path) map parseRating).cache()
  }

  def getDataDesc(spark: SparkSession,path: String) = {
    fromHdfsToLoadData(spark,path).show()
    fromHdfsToLoadData(spark,path).describe("userId","movieId","rating")show()
  }

}
