package spark.mllib.classification

import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.{Row, SparkSession}

object DTClassPredictDemo extends  Serializable {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder()
      .master("local[4]")
      .appName("DTClassPredictDemo")
      .config("spark.testing.memory", "1073741824")
      .getOrCreate()

    Logger.getLogger("org").setLevel(Level.WARN)
    val filePath = "data/classnews/predict"

    val preprocessor = new Preprocessor
    val(predictDF,indexModel,_) = preprocessor.predict(filePath,spark)

    val dtClassifier = new DTClassifier
    val predictions = dtClassifier.perdict(predictDF,indexModel)

    val resultRDD = predictions
      .select("prediction","indexedLabel")
      .rdd
      .map { case Row(prediction : Double, label : Double) =>
        (prediction,label)}

    val (precision, recall,f1) = Evaluations.multiClassEvaluate(resultRDD)

    println("\n\n======= 评估结果 ======")
    println(s"加权准确率: $precision \n")
    println(s"加权召回率: $recall \n")
    println(s"F1 值: $f1")

    predictions.select("label","predictedLabel","content").show(100,truncate = true)

    spark.stop()

  }
}
