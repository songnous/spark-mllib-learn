package spark.mllib.classification

import org.apache.spark.sql.{DataFrame, SparkSession}

object newsLR {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder()
      .appName("newsLR")
      .getOrCreate()

  }

  def clean(filePath : String, spark : SparkSession) : DataFrame = {
    import spark.implicits._
    val textDF = spark.sparkContext.textFile(filePath).flatMap{line =>
      val fields = line.split("\u00ef")
      if(fields.length > 3) {
        val categoryLine = fields(0)
        val categories = categoryLine.split("\\|")
        val category = categories.last
        var label = -1.0
        if(category.contains("文化")) label = 0.0
        else if(category.contains("财经")) label = 1.0
        else if(category.contains("军事")) label = 3.0
        else {}

        val title = fields(1)
        val time = fields(2)
        val content = fields(3)
        if(label > -1) Some(label,title,time,content) else None
      } else None
    }.toDF("label","title","time","content")
    textDF
  }
}
