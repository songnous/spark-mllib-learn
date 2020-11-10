package spark.mllib.features

import org.apache.spark.ml.feature.MinMaxScaler
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.SparkSession

object minMaxScalerExample {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder()
      .master("local")
      .appName("minMaxScalerExample")
      .config("spark.testing.memory", "471859200")
      .getOrCreate()

    val dataFrame = spark.createDataFrame(Seq(
      (0, Vectors.dense(1.0,0.1,-1.0)),
      (1, Vectors.dense(2.0,1.1,1.0)),
      (2, Vectors.dense(3.0,10.0,3.0))
    )).toDF("id", "spark/mllib/features")

    val scaler = new MinMaxScaler()
      .setInputCol("spark/mllib/features")
      .setOutputCol("scaledFeatures")

    // 计算汇总统计并生成 MinMaxScalerModel
    val scalerModel = scaler.fit(dataFrame)

    // 将每个特征重新缩放至 {min,max} 范围

    val scaledData = scalerModel.transform(dataFrame)

    println(s"Features scaled to range: [${scaler.getMin}],${scaler.getMax}")
    scaledData.select("spark/mllib/features","scaledFeatures").show()
  }
}
