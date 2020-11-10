package spark.mllib.classification

import org.apache.spark.sql.SparkSession

object LRClassTrainDemo {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder()
      .master("local[4]")
      .appName("LRClassTrainDemo")
      .config("spark.testing.memory", "1073741824")
      .getOrCreate()
    val filePath = "data/classnews/train"
    // 预处理(清洗、标签索引优化、分词，向量化)
    val preprocessor = new Preprocessor()
    val trainDF = preprocessor.predict(filePath,spark)._1
    // train model
    val lrClassifier = new LRClassifier
    lrClassifier.train(trainDF)

    spark.stop()
  }
}
