package spark.mllib.features

import org.apache.spark.ml.feature.Normalizer
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.SparkSession

object normalizerExample {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder()
      .master("local")
      .appName("normalizerExample")
      .config("spark.testing.memory", "471859200")
      .getOrCreate()

    val dataFrame = spark.createDataFrame(Seq(
      (0,Vectors.dense(1.0,0.5,-1.0)),
      (1,Vectors.dense(2.0,1.0,1.0)),
      (2,Vectors.dense(4.0,10.0,2.0)),
      (3,Vectors.dense(6.0,3.0,-1.0)),
      (4,Vectors.dense(9.0,2.0,8.0)),
      (5,Vectors.dense(7.0,6.0,3.0))
    )).toDF("id", "spark/mllib/features")

    //  使用 L1 正则化向量
    val normalizer1 = new Normalizer()
      .setInputCol("spark/mllib/features")
      .setOutputCol("normFeatures-1")
      .setP(1.0)

    val l1NormData = normalizer1.transform(dataFrame)

    println("Normalized using L1 norm: ")
    l1NormData.show()

    // 使用 L ∞ 正则化向量

    val lInfNormData = normalizer1.transform(dataFrame,normalizer1.p -> Double.PositiveInfinity)
    println("Normalized using L inf norm: ")
    lInfNormData.show()

    // 使用 L2 正则化

    val normalizer2 = new Normalizer()
      .setInputCol("spark/mllib/features")
      .setOutputCol("normFeatures-2")
      .setP(2.0)

    val l2NormData = normalizer2.transform(dataFrame)
    println("Normalized using L2 norm: ")
    l2NormData.show()
  }
}
