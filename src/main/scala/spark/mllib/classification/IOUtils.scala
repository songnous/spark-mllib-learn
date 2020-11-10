package spark.mllib.classification

import java.io.File


object IOUtils {
  def main(args: Array[String]): Unit = {
    val file = new File("E:\\workspace\\idea\\spark-wordcount-idea\\models\\preprocession")
    delDir(file)
  }

  def delDir(file : File) : Boolean ={
    if(file.isDirectory) {
      val subFileList = file.listFiles()
      for(subFile <- subFileList) {
        println("subFile: " + subFileList.mkString("#"))
        delDir(subFile)
      }
      file.delete()
    } else {
      file.delete()
    }
  }
}
