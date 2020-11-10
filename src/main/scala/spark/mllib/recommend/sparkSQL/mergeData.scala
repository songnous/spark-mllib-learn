package spark.mllib.recommend.sparkSQL

import org.apache.spark.sql.Row
import org.apache.spark.sql.types._
import scala.collection.mutable
import org.apache.spark.{SparkConf,SparkContext}

object mergeData {

    def main(args: Array[String]) {
        
        // create talble schema of the userprofile
        val userProfileSchema: StructType = StructType(mutable.ArraySeq(
            StructField("userid",StringType,nullable = false),
            StructField("gender",StringType,nullable = false),
            StructField("age",StringType,nullable = false),
            StructField("salary",StringType,nullable = false),
            StructField("userlocation",StringType,nullable = false)
        ))


        // create talble schema of the userwatch
        val userWatchSchema: StructType = StructType(mutable.ArraySeq(
            StructField("userid",StringType,nullable = false),
            StructField("itemid",StringType,nullable = false),
            StructField("watchlen",IntegerType,nullable = false),
            StructField("hour",IntegerType,nullable = false)
        ))


        // create talble schema of the musicmeta
        val musicMetaSchema: StructType = StructType(mutable.ArraySeq(
            StructField("itemid",StringType,nullable = false),
            StructField("name",StringType,nullable = false),
            StructField("desc",StringType,nullable = false),
            StructField("total_timelen",IntegerType,nullable = false),
            StructField("itemlocation",StringType,nullable = false),
            StructField("tags",StringType,nullable = false)
        ))

        val conf = new SparkConf().setAppName("merge data")
        val sc = new SparkContext(conf)
        val sqlContext = new org.apache.spark.sql.SQLContext(sc)

        val userProfileDataFile = args(0)
        val userWatchDataFile = args(1)
        val musicMetaDataFile = args(2)
        val output = args(3)

        // gen data of userprofile table
        val userProfileData = sc.textFile(userProfileDataFile).filter {lines => 
            val line = lines.trim().split(",")
            line.length == 5
        }.map {lines =>
            val line = lines.trim().split(",")
            Row(line(0),line(1),line(2),line(3),line(4))
        }


        // gen data of userwatch table
        val userWatchData = sc.textFile(userWatchDataFile).filter {lines => 
            val line = lines.trim().split("")
            line.length == 4
        }.map {lines =>
            val line = lines.trim().split("")
            Row(line(0),line(1),line(2).toInt,line(3).toInt)
        }
        
       // gen data of musicmeta table
       val musicMetaData = sc.textFile(musicMetaDataFile).filter {lines => 
           val line = lines.trim().split("")
           line.length == 6
       }.map {lines =>
           val line = lines.trim().split("")
           Row(line(0),line(1),line(2),line(3).toInt,line(4),line(5))
       }

        // create table of userprofile
        val userProfileTable = sqlContext.createDataFrame(userProfileData,userProfileSchema)
        userProfileTable.registerTempTable("userprofile")

        // create table of userwatch
        val userWatchTable = sqlContext.createDataFrame(userWatchData,userWatchSchema)
        userWatchTable.registerTempTable("userwatch")


        // create table of musicmeta
        val musicMetaTable = sqlContext.createDataFrame(musicMetaData,musicMetaSchema)
        musicMetaTable.registerTempTable("musicmeta")

        // SELECT COMMON
        val selectCmd = """SELECT uw.userid,uw.itemid,uw.watchlen,uw.hour,
                           u.gender,u.age,u.salary,u.userlocation,
                           i.name,i.desc,i.total_timelen,i.itemlocation,i.tags
                           FROM userwatch uw
                           JOIN userprofile u ON u.userid = uw.userid
                           JOIN musicmeta i ON i.itemid = uw.itemid
                        """
        // executor the query to save results to HDFS
        // sqlContext.sql(selectCmd).write.format("com.databricks.spark.csv").save(output)
        sqlContext.sql(selectCmd).write.format("csv").save(output)
    }
}
