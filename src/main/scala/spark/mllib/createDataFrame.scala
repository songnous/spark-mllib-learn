package spark.mllib

import org.apache.spark.sql.SparkSession

object createDataFrame {
  def main(args: Array[String]) {
    val spark = SparkSession.builder()
      .appName("Spark SQL basic example")
      .getOrCreate()

    val input = args(0)
    val df1 = spark.read.option("header",true).format("csv").load(input)

    val df2 = df1.select(
      df1("name").cast("String"),
      df1("age").cast("Double"),
      df1("gender").cast("String"))

    df2.printSchema()

    df2.createOrReplaceTempView("custormer")

    val cust1 = spark.sql("SELECT * FROM custormer WHERE age BETWEEN 30 AND 35")
    cust1.limit(5).show()
    cust1.show(5)

    val cust2 = spark.sql("SELECT * FROM custormer WHERE gender like 'M'")
    cust2.limit(5).show()
    cust2.show(5)
  }
}
