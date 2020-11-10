package spark.mllib.classification


import java.io.{File, FileInputStream, InputStreamReader}
import java.util.Properties

import scala.collection.mutable

object Conf extends Serializable {
  def loadConf(filePath : String) :  mutable.LinkedHashMap[String, String] = {
    val kvMap = mutable.LinkedHashMap[String,String]()
    val properties = new Properties()
    properties.load(new InputStreamReader(new FileInputStream(filePath),"UTF-8"))
    val propertyNameArray = properties.stringPropertyNames().toArray(new Array[String](0))

    val fileName = new File(filePath).getName

    println(s"============= 加载配置文件$fileName ================")
    for (propertyName <- propertyNameArray) {
      val property = properties.getProperty(propertyName).replaceAll("\"","").trim
      println(propertyName + ":" + property)
      kvMap.put(propertyName, property)
    }
    println("=================================")
    kvMap
  }
}