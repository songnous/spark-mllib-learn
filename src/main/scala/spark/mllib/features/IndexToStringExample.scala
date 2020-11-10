package spark.mllib.features

import org.apache.spark.ml.attribute.Attribute
import org.apache.spark.ml.feature.{IndexToString, StringIndexer}
import org.apache.spark.sql.SparkSession

object IndexToStringExample {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder()
      .master("local")
      .appName("IndexToStringExample")
      .config("spark.testing.memory", "471859200")
      .getOrCreate()

    val df = spark
      .createDataFrame(
        Seq(
          (0, "a"),
          (1, "b"),
          (2, "c"),
          (3, "a"),
          (4, "a"),
          (5, "c")
        ))
      .toDF("id", "category")

    val indexer = new StringIndexer()
      .setInputCol("category")
      .setOutputCol("categoryIndex")
      .fit(df)

    val indexed = indexer.transform(df)

    println(
      s"Transformed string column '${indexer.getOutputCol}'" + s"to indexed column '${indexer.getOutputCol}'")

    val inputcolSchema = indexed.schema(indexer.getOutputCol)

    println(
      s"StringIndexer will store labels in output column metadata: " + s"${Attribute.fromStructField(inputcolSchema).toString}\n")

    val converter = new IndexToString()
      .setInputCol("categoryIndex")
      .setOutputCol("originalCategory")

    val converted = converter.transform(indexed)

    println(
      s"Transformed indexed column '${converter.getInputCol}' back to original string" + s"column '${converter.getOutputCol}' using lables in metadata")

    converted.select("id","categoryIndex","originalCategory").show()

  }
}
