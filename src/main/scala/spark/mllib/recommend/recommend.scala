package spark.mllib.recommend

import org.apache.spark.sql.SparkSession

object recommend {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder()
      .appName("transformJsonFormat")
      .getOrCreate()
  }

}
