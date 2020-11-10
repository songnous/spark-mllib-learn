package spark.mllib.features

import org.apache.spark.ml.feature.{HashingTF, IDF, Tokenizer}
import org.apache.spark.sql.SparkSession

object tfIDFExample {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder()
      .master("local")
      .appName("TF-IDF Example")
      .config("spark.testing.memory", "471859200")
      .getOrCreate()

    val sentenceData = spark
      .createDataFrame(
        Seq(
          (0, "Hi I heard about Spark"),
          (0, "I wish Java could use case classes"),
          (1, "Logistic regression models are neat")
        ))
      .toDF("label", "sentences")

    val tokenizer = new Tokenizer()
      .setInputCol("sentences")
      .setOutputCol("words")

    val wordData = tokenizer.transform(sentenceData)

    val hashingTF = new HashingTF()
      .setInputCol("words")
      .setOutputCol("rawFeatures")
      .setNumFeatures(20)

    val featurizedData = hashingTF.transform(wordData)
    // CountVectorizer 也能获取词频向量
    val idf = new IDF()
      .setInputCol("rawFeatures")
      .setOutputCol("spark/mllib/features")

    val idfModel = idf.fit(featurizedData)

    val rescaleData = idfModel.transform(featurizedData)

    rescaleData.select("spark/mllib/features", "label").take(3).foreach(println)
  }
}
