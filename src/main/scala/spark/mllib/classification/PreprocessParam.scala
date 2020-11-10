package spark.mllib.classification

import scala.collection.mutable

class PreprocessParam extends  Serializable {
  /*
  * kvMap.getOrElse("key1","value1") 如果所查询的 key1 存在于此 Map 中，就输出 key1 对应的 value_old 值，如果 key1 不存在于此 Map 中，就输出 value 的值，
  * Map 保持不变，并不会多一个 key1
  * 这样就不用判断此 key 在 Map 是否存在。
  * kvMap : scala.collection.mutable.LinkedHashMap[String,String] = Map(key1 -> 85)
  * key1 存在，所以输出 key1 对应的 value_old： 85
  * kvMap.getOrElse("key1","10") 输出：String = 85
  * key2 不存在，所以直接输出给定的 value： 10
  * kvMap.getOrElse("key2","10") 输出：String = 10
  * */
  val kvMap : mutable.LinkedHashMap[String,String] = Conf.loadConf("src/main/resources/preprocess.properties")
  val stopWordFilePath : String = kvMap.getOrElse("stopword.file.path","src/main/data/stopwords.txt")
  val segmenterType : String = kvMap.getOrElse("segment.type","StandardSegment")
  val delNum : Boolean = kvMap.getOrElse("is.delete.number","false").toBoolean
  val delEn : Boolean = kvMap.getOrElse("is.delete.english","false").toBoolean
  val addNature : Boolean = kvMap.getOrElse("add.nature","false").toBoolean
  val minTermLen : Int = kvMap.getOrElse("min.term.len","1").toInt
  val minTermNum : Int = kvMap.getOrElse("min.term.num","3").toInt
  val vocabSize : Int  = kvMap.getOrElse("vocab.size","10000").toInt

  val IndexModelPath : String = kvMap.getOrElse("model.index.path","data/model/preprocession/indexMldel")
  val vecModelPath : String = kvMap.getOrElse("model.vectorize.path","data/model/preprocession/VecModel")
}
