package spark.mllib.classification

import java.io.File

import org.apache.spark.ml.classification.{DecisionTreeClassificationModel, DecisionTreeClassifier}
import org.apache.spark.ml.feature.{IndexToString, StringIndexerModel}
import org.apache.spark.sql.DataFrame

class DTClassifier extends Serializable {

  def saveModel(dtModel: DecisionTreeClassificationModel, params: ClassParam) : Unit = {
    val filePath = params.modelDTPath
    val file = new File(filePath)
    println("========================")
    println("file path: " + file)
    if(file.exists()) {
      println("决策树模型已存在，新模型将覆盖原有的模型!")
      IOUtils.delDir(file)
    }
    dtModel.save(filePath)
  }

  def train(data : DataFrame) : DecisionTreeClassificationModel = {
    val params = new ClassParam

    data.persist()

    val dtModel = new DecisionTreeClassifier()
      .setMinInfoGain(params.minInfoGain)
      .setMaxDepth(params.maxDepth)
      .setLabelCol("indexedLabel")
      .setFeaturesCol("spark/mllib/features")
      .fit(data)

    data.unpersist()
    this.saveModel(dtModel,params)
    dtModel
  }

  def loadModel(params : ClassParam) : DecisionTreeClassificationModel = {
    val filePath = params.modelDTPath
    val file = new File(filePath)
    if(! file.exists()) {
      println("决策树模型不存在，即将退出！")
      System.exit(1)
    } else {
      println("开始加载决策树模型....")
    }

    val dtModel = DecisionTreeClassificationModel.load(filePath)
    println("决策树模型加载成功...")

    dtModel
  }

  def perdict(data : DataFrame,indexModel : StringIndexerModel) : DataFrame = {
    val params = new ClassParam
    val dtModel = this.loadModel(params)

    val predictions = dtModel.transform(data)

    val labelConverter = new IndexToString()
      .setInputCol("prediction")
      .setOutputCol("predictedLabel")
      .setLabels(indexModel.labels)

    val result = labelConverter.transform(predictions)

    result
  }
}
