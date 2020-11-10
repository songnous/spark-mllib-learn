package spark.mllib.classification

import java.io.File

import org.apache.spark.ml.classification.{LogisticRegression, LogisticRegressionModel}
import org.apache.spark.ml.feature.{IndexToString, StringIndexerModel}
import org.apache.spark.sql.DataFrame

class LRClassifier extends Serializable {

  def saveModel(lrModel: LogisticRegressionModel, params: ClassParam): Unit = {
    val filePath = params.modelLRPath
    val file = new File(filePath)
    if(file.exists()) {
      println("LR 模型已存在，新模型将覆盖原有模型")
      IOUtils.delDir(file)
    }

    lrModel.save(filePath)
    println("LR 模型已保存")
  }

  def train(data: DataFrame) : LogisticRegressionModel = {
    val params = new ClassParam

    data.persist()
    val lrModel = new LogisticRegression()
      .setMaxIter(params.maxIteration)
      .setRegParam(params.regParam)
      .setElasticNetParam(params.elasticNetParam)
      .setTol(params.converTol)
      .setLabelCol("indexedLabel")
      .setFeaturesCol("spark/mllib/features")
      .fit(data)
    data.unpersist()
    this.saveModel(lrModel,params)

    lrModel
  }
  def loadModel(params: ClassParam) : LogisticRegressionModel = {
   val filePath = params.modelLRPath
    val file = new File(filePath)
    if(! file.exists()) {
      println("LR模型不存在，即将退出！")
      System.exit(1)
    } else {
      println("开始加载 LR 模型")
    }

    val lrModel = LogisticRegressionModel.load(filePath)
    println("LR 模型加载成功")

    lrModel
  }

  def predict(data: DataFrame,indexModel : StringIndexerModel) : DataFrame = {
    val params = new ClassParam
    val lrModel = this.loadModel(params)

    val predictions = lrModel.transform(data)

    val labelConverter = new IndexToString()
      .setInputCol("prediction")
      .setOutputCol("predictedLabel")
      .setLabels(indexModel.labels)

    val result = labelConverter.transform(predictions)
    result
  }
}
