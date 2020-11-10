package spark.mllib.features

import org.apache.spark.ml.feature.Bucketizer
import org.apache.spark.sql.SparkSession

object bucketizerExample {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder()
      .master("local")
      .appName("bucketizerExample")
      .config("spark.testing.memory", "471859200")
      .getOrCreate()

    val split = Array(Double.NegativeInfinity,-0.5,0.0,0.5,Double.PositiveInfinity)

    val data = Array(-999.9,-0.5,-0.3,0.0,0.2,999.9)

    val dataFrame = spark.createDataFrame(
      data.map(Tuple1.apply)
    ).toDF("spark/mllib/features")

    val bucketizer = new Bucketizer()
      .setInputCol("spark/mllib/features")
      .setOutputCol("bucketedFeatures")
      .setSplits(split)

    // 将原来的数值转换为箱式索引
    val bucketedData = bucketizer.transform(dataFrame)

    println(s"Bucketizer output with ${bucketizer.getSplits.length - 1} buckets")

    bucketedData.show()
  }
}
