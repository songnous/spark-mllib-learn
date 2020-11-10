package spark.mllib.features

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.{HashingTF, Tokenizer}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.{Row, SparkSession}

object pipeLineExample1 {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder()
      .master("local")
      .appName("pipeLineExample2")
      .config("spark.testing.memory", "471859200")
      .getOrCreate()
    val training = spark
      .createDataFrame(
        Seq(
          (0L, "a b c d e spark", 1.0),
          (1L, "b d", 0.0),
          (2L, "spark f g h", 1.0),
          (3L, "hadoop mapreduce", 0.0)
        ))
      .toDF("id", "text", "label")

    val tokenizer = new Tokenizer()
      .setInputCol("text")
      .setOutputCol("words")
    val hashingTF = new HashingTF()
      .setNumFeatures(1000)
      .setInputCol(tokenizer.getOutputCol)
      .setOutputCol("spark/mllib/features")
    val lr = new LogisticRegression()
      .setMaxIter(10)
      .setRegParam(0.001)

    val pipeline = new Pipeline()
      .setStages(Array(tokenizer, hashingTF, lr))

    val model = pipeline.fit(training)

    val output = args(0)
    val output2 = args(1)
    model.write.overwrite().save(output)

    pipeline.write.overwrite().save(output2)

    val test = spark
      .createDataFrame(
        Seq(
          (4L, "spark i j k"),
          (5L, "l m n"),
          (6L, "spark hadoop spark"),
          (7L, "apache hadoop")
        ))
      .toDF("id", "text")

    model
      .transform(test)
      .select("id", "text", "probability", "prediction")
      .collect()
      .foreach {
        case Row(id: Long, text: String, prob: Vector, prediction: Double) =>
          println(s"($id,$text) --> prob=$prob,prediction=$prediction")
      }
  }
}
