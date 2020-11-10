package spark.mllib.features

import org.apache.spark.ml.feature.QuantileDiscretizer
import org.apache.spark.sql.SparkSession

object quantileDisretizerExample {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder()
      .master("local")
      .appName("quantileDisretizerExample")
      .config("spark.testing.memory", "471859200")
      .getOrCreate()

    val data = Array((0,18.0),(1,19.0),(2,8.0),(3,5.0),(4,2.2))

    val df = spark.createDataFrame(data).toDF("id","hour")

    val discretizer = new QuantileDiscretizer()
      .setInputCol("hour")
      .setOutputCol("result")
      .setNumBuckets(3)

    val result = discretizer.fit(df).transform(df)

    result.show()
  }
}
