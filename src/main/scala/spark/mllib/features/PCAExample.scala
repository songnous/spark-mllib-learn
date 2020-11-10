package spark.mllib.features

import org.apache.spark.ml.feature.{PCA, StandardScaler}
import org.apache.spark.sql.SparkSession

object PCAExample {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder()
      .master("local")
      .appName("PCAExample")
      .config("spark.testing.memory", "471859200")
      .getOrCreate()
    val input = args(0)
    val rawDataFrame = spark.read.format("libsvm").load(input)

    val scaleDataFrame = new StandardScaler()
      .setInputCol("spark/mllib/features")
      .setOutputCol("scaledFeatures")
      .setWithMean(false) // 对于稀疏数据(如本次使用的数据)，不要使用平均值
      .setWithStd(true)
      .fit(rawDataFrame)
      .transform(rawDataFrame)

    // PCA 模型
    val pcaModel = new PCA()
      .setInputCol("scaledFeatures")
      .setOutputCol("pcaFeatures")
      .setK(3)
      .fit(scaleDataFrame)

    // PCA 降维

    pcaModel.transform(scaleDataFrame)
      .select("label","pcaFeatures")
      .show(10,false)
    // 没有标准化特征向量，直接进行PCA 主成分分析：各主成分之间值变化太大，有数量级的差别。
    // 标准化特征向量后 PCA 主成分分析，各主成分之间的值基本上在同一个水平，结果更合理

    // 如何选择 k 值

    val pcaModel1 =new PCA()
      .setInputCol("scaledFeatures")
      .setOutputCol("pcaFeatures")
      .setK(50)
      .fit(scaleDataFrame)

    var i = 1
    for ( x <- pcaModel1.explainedVariance.toArray) {
      println(i + "\t" + x + " ")
      i += 1
    }

  }
}
