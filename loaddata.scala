package spark.mllib

import org.apache.spark.sql.Row
import org.apache.spark.sql.types._
import scala.collection.mutable
import org.apache.spark.{SparkConf,SparkContext}

object loadData {

    def main(args: Array[String]) {

        val userSchema: StructType = StructType(mutable.ArraySeq(
            StructField("userid",IntegerType,nullable = false),
            StructField("age",IntegerType,nullable = false),
            StructField("gender",StringType,nullable = false),
            StructField("occupation",StringType,nullable = false),
            StructField("zipcode",StringType,nullable = false)
        ))
        val conf = new SparkConf().setAppName("load data")
        val sc = new SparkContext(conf)
        val sqlContext = new org.apache.spark.sql.SQLContext(sc)
        val userData = sc.textFile("hdfs://master:9000/u01/bigdata/user_test.txt").map {
            lines =>
            val line = lines.split(",")
            Row(line(0).toInt,line(1)toInt,line(2),line(3),line(4))
        }
        val userTable = sqlContext.createDataFrame(userData,userSchema)
        userTable.registerTempTable("user")
        sqlContext.sql("SELECT max(userid) as useridMax FROM user").show()
        
    }
}
