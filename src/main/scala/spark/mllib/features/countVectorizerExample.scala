package spark.mllib.features

import org.apache.spark.ml.feature.{CountVectorizer, CountVectorizerModel}
import org.apache.spark.sql.SparkSession

object countVectorizerExample {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder()
      .master("local")
      .appName("countVectorizerExample")
      .config("spark.testing.memory", "471859200")
      .getOrCreate()

    val df = spark.createDataFrame(Seq(
      (0,Array("a","b","c")),
      (1,Array("a","d","b","m","n"))
    )).toDF("id","words")

    // 从语料库中拟合 CountVectorizerModel
    val cvModel:CountVectorizerModel = new CountVectorizer()
      .setInputCol("words")
      .setOutputCol("spark/mllib/features")
      .setVocabSize(3)
      .setMinDF(2)
      .fit(df)

    //  也可以用先验证词汇表定义 CountVectorizerModel

    val cvm = new CountVectorizerModel(Array("a","b","c"))
      .setInputCol("words")
      .setOutputCol("spark/mllib/features")

    cvModel.transform(df).show(false)
  }
}
