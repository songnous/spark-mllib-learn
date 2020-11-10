package spark.mllib.features

import org.apache.spark.ml.feature.MaxAbsScaler
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.SparkSession

object maxAbsScalerExample {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder()
      .master("local")
      .appName("maxAbsScalerExample")
      .config("spark.testing.memory", "471859200")
      .getOrCreate()

    val dataFrame = spark.createDataFrame(Seq(
      (0,Vectors.dense(1.0,0.1,-8.0)),
      (1,Vectors.dense(2.0,1.0,-4.0)),
      (2,Vectors.dense(4.0,10.0,8.0))
    )).toDF("id", "spark/mllib/features")

    val scaler = new MaxAbsScaler()
      .setInputCol("spark/mllib/features")
      .setOutputCol("scaledFeatures")

    // 计算汇总并生成 MaxAbsScalerModel 模型
    val scalerModel = scaler.fit(dataFrame)
    val scaledData = scalerModel.transform(dataFrame)
    scaledData.select("spark/mllib/features","scaledFeatures").show()
  }
}
