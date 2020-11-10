package spark.mllib.features

import org.apache.spark.ml.feature.VectorIndexer
import org.apache.spark.sql.SparkSession

object vectorIndexerExample {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder()
      .master("local")
      .appName("vectorIndexerExample")
      .config("spark.testing.memory", "471859200")
      .getOrCreate()

    val input = args(0)
    val data = spark.read.format("libsvm").load(input)

    val indexer = new VectorIndexer()
      .setInputCol("spark/mllib/features")
      .setOutputCol("indexed")
      .setMaxCategories(10)

    val indexerModel = indexer.fit(data)

    val catagoricalFeatures : Set[Int] = indexerModel.categoryMaps.keys.toSet
    println(s"Chose ${catagoricalFeatures.size} categroical features: " +
    catagoricalFeatures.mkString(", "))

    // 创建一个新列，"索引" 列，将分类特征转换为索引
    val indexedData = indexerModel.transform(data)

    indexedData.show()

  }
}
