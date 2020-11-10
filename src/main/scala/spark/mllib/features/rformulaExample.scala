package spark.mllib.features

import org.apache.spark.ml.feature.RFormula
import org.apache.spark.sql.SparkSession

object rformulaExample {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder()
      .master("local")
      .appName("rformulaExample")
      .config("spark.testing.memory", "471859200")
      .getOrCreate()

    val dataSet = spark.createDataFrame(Seq(
      (7,"US",18,1.0),
      (8,"CA",12,0.0),
      (9,"NZ",15,0.0)
    )).toDF("id","country","hour","clicked")

    val formula = new RFormula()
      .setFormula("clicked ~ country + hour")
      .setFeaturesCol("spark/mllib/features")
      .setLabelCol("label")

    val output = formula.fit(dataSet).transform(dataSet)
    output.select("spark/mllib/features","label").show()
  }
}
