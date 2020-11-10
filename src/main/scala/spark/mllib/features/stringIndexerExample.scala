package spark.mllib.features

import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.sql.SparkSession

object stringIndexerExample {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder()
      .master("local")
      .appName("stringIndexerExample")
      .config("spark.testing.memory", "471859200")
      .getOrCreate()


    val df = spark.createDataFrame(Seq (
      (0,"a"),
      (1,"b"),
      (2,"c"),
      (3,"a"),
      (4,"a"),
      (5,"c")
    )).toDF("id","category")


    val indexer = new StringIndexer()
      .setInputCol("category")
      .setOutputCol("categoryIndexer")

    val indexed = indexer.fit(df).transform(df)
    println("source data: ")
    df.show()

    println("StringIndexer: ")
    indexed.show()

  }
}
