package spark.mllib.features

import org.apache.spark.ml.attribute.Attribute
import org.apache.spark.ml.feature.{IndexToString, StringIndexer}
import org.apache.spark.sql.SparkSession

object oneHotEncoderEstimatorExample {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder()
      .master("local")
      .appName("oneHotEncoderEstimatorExample")
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
      .setOutputCol("cateoryIndex")
      .fit(df)

    val indexed = indexer.transform(df)

    println(
      s"Thransformed string column '${indexer.getOutputCol}'" + s"to indexed column '${indexer.getOutputCol}'")

    indexed.show()

    val inputColschema = indexed.schema(indexer.getOutputCol)
    println(
      s"StringIndexer will store lables in output column metadata: " +
        s"${Attribute.fromStructField(inputColschema).toString}\n")

    val converter = new IndexToString()
      .setInputCol("cateoryIndex")
      .setOutputCol("orgiginalCategory")

    val converted = converter.transform(indexed)

    println(
      s"Transformed indexed column '${converter.getOutputCol}' back to original string" +
        s"column '${converter.getOutputCol}' using labels in metadata")
    converted.select("id","cateoryIndex","orgiginalCategory").show()
  }
}
