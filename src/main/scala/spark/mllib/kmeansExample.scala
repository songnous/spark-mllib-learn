package spark.mllib

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.feature.{OneHotEncoder, StandardScaler, VectorAssembler}
import org.apache.spark.sql.SparkSession

object kmeansExample {
  def main(args : Array[String]) {
    val spark = SparkSession.builder().appName("kmeansExample").master("yarn").getOrCreate()

    val rawData = spark.read.format("csv").option("header",true).load(args(0))
    rawData.show(3)
    println(rawData.printSchema())

    val data1 = rawData.select(
      rawData("Channel").cast("Double"),
      rawData("Region").cast("Double"),
      rawData("Fresh").cast("Double"),
      rawData("Milk").cast("Double"),
      rawData("Grocery").cast("Double"),
      rawData("Frozen").cast("Double"),
      rawData("Detergents_Paper").cast("Double"),
      rawData("Delicassen").cast("Double")
    ).cache()

    data1.select("Fresh","Milk","Grocery","Frozen","Detergents_Paper","Delicassen").describe().show()

    val datahost1 = new OneHotEncoder().setInputCol("Channel").setOutputCol("ChannelVector").setDropLast(false)
    val datahost2 = new OneHotEncoder().setInputCol("Region").setOutputCol("RegionVector").setDropLast(false)
    val featuresArray = Array("ChannelVector","RegionVector","Fresh","Milk","Grocery","Frozen","Detergents_Paper","Delicassen")

    val vecDF = new VectorAssembler().setInputCols(featuresArray).setOutputCol("spark/mllib/features")

    val scaleDF = new StandardScaler().setInputCol("spark/mllib/features").setOutputCol("scaledFeatures").setWithStd(true).setWithMean(false)

    val kmeans = new KMeans().setFeaturesCol("scaledFeatures").setK(4).setSeed(123)

    val pipeline1 = new Pipeline().setStages(Array(datahost1,datahost2,vecDF,scaleDF))

    val data2 = pipeline1.fit(data1).transform(data1)

    val model = kmeans.fit(data2)
    val results = model.transform(data2)

    val WSSSE = model.transform(data2)
    println(s"Within Set Sum of Squared Errors = $WSSSE")
    println("cluster centers: ")
    model.clusterCenters.foreach(println)
    results.collect().foreach(row => {println( row(10) + " is predicted as cluster " + row(11))})
    results.select("scaledFeatures","prediction").groupBy("prediction").count.show()
    results.select("scaledFeatures","prediction").filter( i => i(1) == 0).show(20)
    val result0 = results.select("scaledFeatures","prediction").filter(i => i(1) == 0).select("scaledFeatures")
    result0.show()
  }
}
