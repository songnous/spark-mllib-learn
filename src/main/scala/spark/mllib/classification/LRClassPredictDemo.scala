package spark.mllib.classification

import org.apache.spark.sql.{Row, SparkSession}

object LRClassPredictDemo {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder()
      .master("local[4]")
      .appName("LRClassPredictDemo")
      .config("spark.testing.memory", "1073741824")
      .getOrCreate()

    val filePath = "data/classnews/predict"
    // 预处理(清洗，分词，向量化)
    val preprocessor = new Preprocessor
    val (predictDF,indexModel,_) = preprocessor.predict(filePath,spark)

    // train model
    val lrClassifier = new LRClassifier
    val predictions = lrClassifier.predict(predictDF,indexModel)

    // 模型评估
    val resultRDD = predictions.select("prediction","indexedLabel").rdd.map{
      case Row(prediction: Double,label: Double) => (prediction,label)
    }
    val (precision,recall,f1) = Evaluations.multiClassEvaluate(resultRDD)

    predictions.select("label","predictedLabel","content").show(100,truncate = false)

    println("\n====评估结果====")
    println(s"加权准确率： $precision")
    println(s"加权召回率： $recall")
    println(s"F1 值： $f1")
    spark.stop()

  }
}
