package spark.mllib.classification

import org.apache.spark.sql.SparkSession

object Native_Bayes_classifier {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder()
      .master("local")
      .appName("Native_Bayes_classifier")
      .config("spark.testing.memory", "471859200")
      .getOrCreate()

    
  }
}
