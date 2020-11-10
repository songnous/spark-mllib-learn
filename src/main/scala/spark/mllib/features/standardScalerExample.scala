package spark.mllib.features

import org.apache.spark.ml.feature.StandardScaler
import org.apache.spark.sql.SparkSession

object standardScalerExample {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder()
      .master("local")
      .appName("standardScalerExample")
      .config("spark.testing.memory", "471859200")
      .getOrCreate()

    val input = args(0)
    val dataFrame = spark.read.format("libsvm").load(input)
    val scaler = new StandardScaler()
      .setInputCol("spark/mllib/features")
      .setOutputCol("scaleFeatrues")
      .setWithStd(true)
      .setWithMean(false)

    val scalerModel = scaler.fit(dataFrame)

    val scaledData = scalerModel.transform(dataFrame)
    scaledData.show()
  }
}
