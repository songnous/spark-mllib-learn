package spark.mllib.features

import org.apache.spark.ml.feature.DCT
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.SparkSession

object dctExample {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder()
      .master("local")
      .appName("dctExample")
      .config("spark.testing.memory", "471859200")
      .getOrCreate()

    val data = Seq(
      Vectors.dense(0.0,1.0,-2.0,3.0),
      Vectors.dense(-1.0,2.0,-4.0,-7.0),
      Vectors.dense(14.0,-2.0,-5.0,1.0)
    )

    val df = spark.createDataFrame(data.map(Tuple1.apply)).toDF("spark/mllib/features")

    val dct = new DCT()
      .setInputCol("spark/mllib/features")
      .setOutputCol("featuresDCT")
      .setInverse(false)

    val dctDF = dct.transform(df)

    dctDF.select("featuresDCT").show(false)
  }
}
