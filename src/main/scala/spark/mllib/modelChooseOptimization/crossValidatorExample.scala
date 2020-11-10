package spark.mllib.modelChooseOptimization

import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.classification.{LogisticRegression, LogisticRegressionModel}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.{HashingTF, Tokenizer}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.sql.{Row, SparkSession}
import org.apache.spark.ml.linalg.Vector

object crossValidatorExample {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder()
      .master("local")
      .appName("crossValidatorExample")
      .config("spark.testing.memory", "471859200")
      .getOrCreate()

    // 生成一个含 id,text,label 的训练数据集
    val training = spark.createDataFrame(Seq(
      (0L,"a b c d e spark",1.0),
      (1L,"b d",0.0),
      (2L,"spark f g h",1.0),
      (3L,"hadoop mapreduces",0.0),
      (4L,"b spark who",1.0),
      (5L,"g d a y",0.0),
      (6L,"spark fly",1.0),
      (7L,"was mapreduce",0.0),
      (8L,"e spark program",1.0),
      (9L,"a e c l",0.0),
      (10L,"spark compile",1.0),
      (11L,"hadoop software",1.0)
    )).toDF("id","text","label")

    // 配置一个流水线，该流水线包含 3 个 Stage:tokenizer、hashingTF 和 lr

    val tokenizer = new Tokenizer()
      .setInputCol("text")
      .setOutputCol("words")

    val hashingTF = new HashingTF()
      .setInputCol(tokenizer.getOutputCol)
      .setOutputCol("spark/mllib/features")

    val lr = new LogisticRegression()
      .setMaxIter(10)

    val pipeline = new Pipeline()
      .setStages(Array(tokenizer,hashingTF,lr))

    // 使用 ParamGridBuilder 构造一个参数网格：

    val paramGid = new ParamGridBuilder()
      .addGrid(hashingTF.numFeatures,Array(10,100,1000))
      .addGrid(lr.regParam,Array(0.1,0.01))
      .build()

    // 将流水线嵌入到 CrossValidator 实例中，这样流水线的任务都可以使用参数网格。
    // BinaryClassificationEvaluator 默认评估指标为 AUC (areaUnderROC)

    val cv = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(new BinaryClassificationEvaluator)
      .setEstimatorParamMaps(paramGid)
      .setNumFolds(2)

    // 通过交叉验证模型，获取最优参数集，并测试模型
    val cvModel = cv.fit(training)

    val test = spark.createDataFrame(Seq(
      (4L,"spark i j k"),
      (5L,"l m n"),
      (6L,"spark mapreduce"),
      (7L,"apache hadoop")
    )).toDF("id","text")

    val predictions = cvModel.transform(test)
    predictions.select("id","text","probability","prediction")
      .collect()
      .foreach { case Row(id: Long,text: String,prob: Vector,prediction: Double) =>
      println(s"($id,$text) --> prob=$prob,prediction=$prediction")}

    // 查看最佳模型中的各参数值：
    val bestModel = cvModel.bestModel.asInstanceOf[PipelineModel]
    val lrModel = bestModel.stages(2).asInstanceOf[LogisticRegressionModel]

    println("lrModel.getRegParam: " + lrModel.getRegParam  + "\n" + "lrModel.numFeatures"  + lrModel.numFeatures)

  }
}
