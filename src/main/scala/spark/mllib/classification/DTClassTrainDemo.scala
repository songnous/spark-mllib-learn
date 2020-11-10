package spark.mllib.classification

import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.SparkSession

object DTClassTrainDemo extends Serializable {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder()
      .master("local[4]")
      .appName("DTClassTrainDemo")
      .config("spark.testing.memory", "1073741824")
      .getOrCreate()

    Logger.getLogger("org").setLevel(Level.WARN)

    val filePath = "data/classnews/train"

    val preprocessor = new Preprocessor
    val trainDF = preprocessor.predict(filePath,spark)._1

    val dtClassifier = new DTClassifier
    dtClassifier.train(trainDF)

    spark.stop()

  }
}
