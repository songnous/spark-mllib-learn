package spark.mllib.features

import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.SparkSession

object vectorAssemblerExample {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder()
      .master("local")
      .appName("vectorAssemblerExample")
      .config("spark.testing.memory", "471859200")
      .getOrCreate()

    val dataset = spark.createDataFrame(Seq(
      (0,18,1.0,Vectors.dense(0,0,10.0,0.5),1.0)
    )).toDF("id","hour","mobile","userFeatures","clicked")

    val assembler = new VectorAssembler()
      .setInputCols(Array("id","hour","mobile","userFeatures","clicked"))
      .setOutputCol("spark/mllib/features")

    val output = assembler.transform(dataset)

    println("Assembled columns 'hour','mobile','userFeatures' to vector column 'features' ")
    output.select("id","hour","userFeatures", "spark/mllib/features","clicked").show(false)
  }
}
