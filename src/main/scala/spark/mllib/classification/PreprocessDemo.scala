package spark.mllib.classification

import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.SparkSession

object PreprocessDemo {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder()
      .master("local")
      .appName("PreprocessDemo")
      .config("spark.testing.memory", "471859200")
      .getOrCreate()

    Logger.getLogger("org").setLevel(Level.WARN)

    val filePath = "data/classnews/train"
    val preprocessor = new Preprocessor
    preprocessor.train(filePath,spark)

    spark.stop()
  }
}
