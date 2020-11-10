package spark.mllib.features

import org.apache.spark.ml.feature.NGram
import org.apache.spark.sql.SparkSession

object ngramExample {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder()
      .master("local")
      .appName("ngramExample")
      .config("spark.testing.memory", "471859200")
      .getOrCreate()

    val wordDataFrame = spark.createDataFrame(Seq(
      (0,Array("Hi","I","heard","about","Spark")),
      (1,Array("I","wish","Java","could","use","case","classes")),
      (2,Array("Logistic","Regression","models","are","nead")),
      (3,Array("programe","language","is","python","java","scala","R"))
    )).toDF("id","words")

    val ngram = new NGram()
      .setN(6)
      .setInputCol("words")
      .setOutputCol("ngrams")

    val ngramDataFrame = ngram.transform(wordDataFrame)

    ngramDataFrame.select("ngrams").show(false)
  }
}
