package spark.mllib.classification

import scala.collection.mutable

class ClassParam extends Serializable {
  val kvMap : mutable.LinkedHashMap[String,String] = Conf.loadConf("src/main/resources/classification.properties")

  val maxIteration : Int = kvMap.getOrElse("max.iteration","110").toInt
  val regParam : Double = kvMap.getOrElse("reg.param","0.3").toDouble
  val elasticNetParam : Double = kvMap.getOrElse("elastic.net.param","0.1").toDouble
  val converTol : Double = kvMap.getOrElse("conver.tol","1E-7").toDouble

  val minInfoGain : Double = kvMap.getOrElse("min.info.gain","0.0").toDouble
  val maxDepth : Int = kvMap.getOrElse("max.depth","30").toInt

  val modelLRPath : String = kvMap.getOrElse("model.lr.Path","data/model/classification/lrModel")
  val modelDTPath : String = kvMap.getOrElse("model.dt.Path","data/model/classification/dtModel")
}