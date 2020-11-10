package spark.mllib.features

import org.apache.spark.ml.feature.ElementwiseProduct
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.SparkSession

object elmentwiseProductExample {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder()
      .master("local")
      .appName("elmentwiseProductExample")
      .config("spark.testing.memory", "471859200")
      .getOrCreate()

    val dataFrame = spark.createDataFrame(Seq(
      ("a",Vectors.dense(1.0,2.0,3.0)),
      ("b",Vectors.dense(4.0,5.0,6.0)),
      ("c",Vectors.dense(7.0,8.0,9.0))
    )).toDF("id","vector")

    val transformingVector = Vectors.dense(0.0,1.0,2.0)

    val transformer = new ElementwiseProduct()
      .setScalingVec(transformingVector)
      .setInputCol("vector")
      .setOutputCol("transformingVector")

    // 创建新列，可批量转换向量
    transformer.transform(dataFrame).show()
  }
}
