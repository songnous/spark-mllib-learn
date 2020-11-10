package spark.mllib.recommend

import org.apache.spark.sql.{DataFrame, SparkSession}

import scala.collection.mutable.ArrayBuffer

class dataPreprocess extends Serializable {

  def aggregationFiveItem(arr : Array[Array[String]]) : ArrayBuffer[Tuple5[String,String,String,String,String]]= {
    val arrBuff = new ArrayBuffer[Tuple5[String,String,String,String,String]]()
    // 0 到倒数第5个数，否则会报数组越界
    for(i <- 0 until (arr.length,5) if i <= arr.length - 5){
      // 将数组的5个元素当成 ArrayBuffer 一个元素
      val t = (arr(i)(1),arr(i+1)(1),arr(i+2)(1),arr(i+3)(1),arr(i+4)(1))
      arrBuff += t
    }
    // arrBuff = ArrayBuffer(("A2WOH395IHGS0T","5.0","1496177056","B0040HNZTW"))
    arrBuff
  }

  def processData(filePath : String,spark : SparkSession) : DataFrame = {
    val df = spark.read.text(filePath)
    import spark.implicits._
    df.map(x => x.toString)
      .map(_.replaceAll("[{\\[\\]}]",""))
      .map(_.split(","))
      .rdd.map(x => x.map(_.split(":")))
      .map(aggregationFiveItem).flatMap(x => x)
      .toDF("UserID","Rating","ReviewTime","Review","MealID")

  }
}
