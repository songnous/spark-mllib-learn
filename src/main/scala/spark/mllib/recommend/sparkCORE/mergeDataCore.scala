package spark.mllib.recommend.sparkCORE

import org.apache.spark.{SparkConf,SparkContext}

object mergeDataCore {

    val conf = new SparkConf().setAppName("mergeDataSparkCore")
    val sc = new SparkContext(conf)
    def main(args: Array[String]) {

        val musicMeta = "/res_system/music_meta" 
        // val musicMeta = args(2)
        val musicFieldSep = "\001"
        val musicFiledNums = 6
        val musicRDD = createRDDFromFile(musicMeta,musicFieldSep,musicFiledNums)

        val musicDataArray = musicRDD.collect
        println("musicDataArray.length",musicDataArray.length)

        //val userProfileData= "/res_system/user_profile.data"
        val userProfileData= args(0)
        val userProfileFieldSep = ","
        val userProfileFiledNums = 5
        val userProfileRDD = createRDDFromFile(userProfileData,userProfileFieldSep,userProfileFiledNums)

    
        //val userWatchData= "/res_system/user_watch_pref.sml"
        val userWatchData= args(1)
        val userWatchFieldSep = "\001"
        val userWatchFiledNums = 4
        val userWatchUseridKeyRDD = createRDDFromFile(userWatchData,userWatchFieldSep,userWatchFiledNums)

        
        val output = args(3)

        val userWatchItemidKeyRDD = sc.textFile(userWatchData).filter {line =>
            val fields = line.replaceAll("\\s","").split(userWatchFieldSep,userWatchFiledNums)
            fields.length == userWatchFiledNums
        }.map {line =>
            val fields = line.replaceAll("\\s","").split(userWatchFieldSep,userWatchFiledNums)
            (fields(1),(fields(0),fields(2),fields(3)))
        }


        val mergeDataRDD1 = userWatchUseridKeyRDD.join(userProfileRDD).map {x =>
            ((x._1,x._2._1(0)),(x._2._1(1),x._2._1(2),x._2._2(0),x._2._2(1),x._2._2(2),x._2._2(3)))
        }
        
        val mergeDataRDD2 = userWatchItemidKeyRDD.join(musicRDD).map {x =>
            ((x._2._1._1,x._1),(x._2._2(0),x._2._2(1),x._2._2(2),x._2._2(3),x._2._2(4)))
        }
        
        val mergeData = mergeDataRDD1.cogroup(mergeDataRDD2).map {x =>
            val userid = x._1._1
            val itemid = x._1._2
            val watchlen = x._2._1.head._1
            val hour = x._2._1.head._2
            val gender = x._2._1.head._3
            val salor = x._2._1.head._4
            val age = x._2._1.head._5
            val userLocation = x._2._1.head._6

            val name = x._2._2.head._1
            val describe = x._2._2.head._2
            val totalLen = x._2._2.head._3
            val itemLocation = x._2._2.head._4
            val tags = x._2._2.head._5
            userid +     "\001" + itemid + "\001" + watchlen + "\001" + hour + "\001" + gender +
            "\001" + salor + "\001" + age + "\001" + userLocation + "\001" + name + "\001" +
            describe + "\001" + totalLen + "\001" + itemLocation  + "\001" + tags
        
        }.saveAsTextFile(output)
    }
    
    def createRDDFromFile(filePath: String,fields_sep: String,fieldNums: Int) : org.apache.spark.rdd.RDD[(String,Array[String])] = {
    
        val rdd = sc.textFile(filePath).filter { line =>
            val fields = line.replaceAll("\\s","").split(fields_sep,fieldNums)
            fields.length == fieldNums
        }.map {line =>
            val fields = line.replaceAll("\\s","").split(fields_sep,fieldNums)
            (fields.head,fields.tail)
        }

        return rdd
    }
        
}
