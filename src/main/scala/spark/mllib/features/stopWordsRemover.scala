package spark.mllib.features

import org.apache.spark.ml.feature.StopWordsRemover
import org.apache.spark.sql.SparkSession

object stopWordsRemover {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder()
      .master("local")
      .appName("stopWordsRemover")
      .config("spark.testing.memory", "471859200")
      .getOrCreate()

    val remove = new StopWordsRemover()
      .setInputCol("raw")
      .setOutputCol("fitlered")

    val dataSet = spark.createDataFrame(Seq(
      (0,Seq("I","saw","the","red","balloon")),
      (1,Seq("Mary","had","a","little","lamb"))
    )).toDF("id","raw")

    remove.transform(dataSet).show(false)
  }
}
