package spark.mllib.features

import org.apache.spark.ml.feature.ChiSqSelector
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.SparkSession

object chiSqSelectorExample {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder()
      .master("local")
      .appName("chiSqSelectorExample")
      .config("spark.testing.memory", "471859200")
      .getOrCreate()

    val data  = Seq(
      (7,Vectors.dense(0.0,0.0,18.0,1.0),1.0),
      (7,Vectors.dense(0.0,1.0,12.0,0.0),0.0),
      (9,Vectors.dense(1.0,0.0,15.0,0.1),0.0)
    )

    val df = spark.createDataFrame(data).toDF("id", "spark/mllib/features","clicked")

    val selector = new ChiSqSelector()
      .setNumTopFeatures(1)
      .setFeaturesCol("spark/mllib/features")
      .setLabelCol("clicked")
      .setOutputCol("selectFeatures")

    val result = selector.fit(df).transform(df)

    println(s"ChiSqSelector output with top ${selector.getNumTopFeatures} features selected")
    result.show()
  }
}
