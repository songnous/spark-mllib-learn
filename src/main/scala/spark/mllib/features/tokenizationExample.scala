package spark.mllib.features

import org.apache.spark.ml.feature.{RegexTokenizer, Tokenizer}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._

object tokenizationExample {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder()
      .master("local")
      .appName("tokenizationExample")
      .config("spark.testing.memory", "471859200")
      .getOrCreate()

    val sentenceDataFrame = spark.createDataFrame(Seq(
      (0,"Hi I header about spark"),
      (1,"I wish Java could use case classes"),
      (0,"Logsitic,regression,models,are,neat")
    )).toDF("id","sentence")

    val tokenizer = new Tokenizer()
      .setInputCol("sentence")
      .setOutputCol("words")

    val regexTokenizer = new RegexTokenizer()
      .setInputCol("sentence")
      .setOutputCol("words")
      .setPattern("\\W") // 或使用 .setPattern("\\W+".setGaps(false)

    val countTokens = udf { (words:Seq[String]) => words.length }

    val tokenized = tokenizer.transform(sentenceDataFrame)
    tokenized.select("sentence","words")
      .withColumn("tokens",countTokens(col("words"))).show(false)

    val regexTokenized = regexTokenizer.transform(sentenceDataFrame)
    regexTokenized.select("sentence","words")
      .withColumn("tokens",countTokens(col("words"))).show(false)
  }
}
