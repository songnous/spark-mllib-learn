package spark.mllib.features

import org.apache.spark.ml.feature.PolynomialExpansion
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.SparkSession

object polynomialExpansionExample {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder()
      .master("local")
      .appName("polynomialExpansionExample")
      .config("spark.testing.memory", "471859200")
      .getOrCreate()

    val data = Array(
      Vectors.dense(2.0, 1.0),
      Vectors.dense(0.0, 0.0),
      Vectors.dense(3.0, -1.0),
      Vectors.dense(5.0,3.0,-6.0,5.0)
    )

    val df = spark.createDataFrame(data.map(Tuple1.apply)).toDF("spark/mllib/features")

    val polyExpansion = new PolynomialExpansion()
      .setInputCol("spark/mllib/features")
      .setOutputCol("polyFeatures")
      .setDegree(3)

    val polyDF = polyExpansion.transform(df)
    polyDF.show(false)
  }
}
