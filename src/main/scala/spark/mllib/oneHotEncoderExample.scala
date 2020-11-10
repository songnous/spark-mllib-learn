package spark.mllib

import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer, VectorAssembler}
import org.apache.spark.sql.{DataFrame, SparkSession}

object oneHotEncoderExample {

  def main(args: Array[String]) {
    val spark = SparkSession
      .builder()
      .master("yarn")
      .appName("oneHotEncoderExample")
      .getOrCreate()

    val df = spark.createDataFrame(Seq(
      (0,"a"),
      (1,"b"),
      (2,"c"),
      (3,"a"),
      (4,"a"),
      (5,"c")
    )).toDF("id","category")

    val indexer = new StringIndexer()
      .setInputCol("category")
      .setOutputCol("categoryIndex")
      .fit(df)

    val indexed = indexer.transform(df)

    val encoder = new OneHotEncoder()
      .setInputCol("categoryIndex")
      .setOutputCol("categoryVec")

    val encoded = encoder.transform(indexed)

    println("endcoded.show(): ")
    encoded.show()


    val fd1 = spark.createDataFrame(Seq(
      (1.0,"a"),
      (1.5,"a"),
      (3.2,"c"),
      (10.1,"b"),
      (3.8,"a"),
      (4.5,"c")
    )).toDF("x","c")

    val ss = new StringIndexer()
      .setInputCol("c")
      .setOutputCol("c_idx")

    val ff = ss.fit(fd1).transform(fd1)
    println("ff.show(): ")
    ff.show()

    val oe = new OneHotEncoder()
      .setInputCol("c_idx")
      .setOutputCol("c_idx_vec")

    val fe = oe.transform(ff)
    println("fe.show(): ")
    fe.show()

    val assembler = new VectorAssembler()
      .setInputCols(Array("x","c_idx","c_idx_vec"))
      .setOutputCol("spark/mllib/features")

    val vecDF: DataFrame = assembler.transform(fe)
    println("vecDF.show(): ")
    vecDF.show()

    val oe1 = new OneHotEncoder()
      .setInputCol("c_idx")
      .setOutputCol("c_idx_vec")
      .setDropLast(false)

    val fe1 = oe1.transform(ff)
    println("fe1.show(): ")
    fe1.show()

    val vecDF1: DataFrame = assembler.transform(fe1)
    println("vecDF1.show(): ")
    vecDF1.show()
  }
}
