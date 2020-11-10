package spark.mllib.features

import org.apache.spark.ml.feature.SQLTransformer
import org.apache.spark.sql.SparkSession

object sqlTransformerExample {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder()
      .master("local")
      .appName("sqlTransformerExample")
      .config("spark.testing.memory", "471859200")
      .getOrCreate()

    val df = spark.createDataFrame(Seq(
      (0,1.0,3.0),
      (2,2.0,5.0),
      (3,3.0,6.0)
    )).toDF("id","v1","v2")

    val sqlTrans = new SQLTransformer()
      .setStatement("SELECT *,(v1+v2) as v3,(v1*v2) as v4 from __THIS__")

    sqlTrans.transform(df).show()
  }
}
