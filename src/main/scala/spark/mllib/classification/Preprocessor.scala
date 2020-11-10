package spark.mllib.classification

import java.io.File

import org.apache.spark.ml.feature._
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.{DataFrame, SparkSession}


class Preprocessor extends Serializable {
  def clean(filePath : String, spark : SparkSession) : DataFrame = {
    /*
    * clean 函数的功能是将训练数据中不属于"文化"，"财经"，"军事"，"体育" 这些类的文章过滤掉
    * */
    import spark.implicits._
    println("==========class Preprocessor=======")
    println("filePath: " + filePath)
    val textDF = spark.sparkContext.textFile(filePath).flatMap{ line =>
      val fields = line.split("\u00ef")

      if(fields.length > 3) {
        val categoryLine = fields(0)
        val categories = categoryLine.split("\\|")
        val category = categories.last

        var label = "其他"
        if(category.contains("文化")) label = "文化"
        else if(category.contains("财经")) label = "财经"
        else if(category.contains("军事")) label = "军事"
        else if(category.contains("体育")) label = "体育"
        else {}

        val title = fields(1)
        val time = fields(2)
        val context = fields(3)
        if (!label.equals("其他")) Some(label, title, context) else None
      } else None
    }.toDF("label","title","content")
    textDF
  }

  def indexrize(data : DataFrame) : StringIndexerModel = {
    // 特征转换，OneHostEncoder 编码
    val labelIndexer = new StringIndexer()
      .setInputCol("label")
      .setOutputCol("indexedLabel")
      .fit(data)

    labelIndexer
  }

  def segment(data : DataFrame, params : PreprocessParam) : DataFrame = {
    val spark = data.sparkSession

    val segmenter = new Segmenter()
      // delEn = flase 删除英文字母
      .isDelEn(params.delEn)
      // delNum = flase 删除数字
      .isDelNum(params.delNum)
      // segmenterType = StandardSegment 分词模式
      .setSegmentType(params.segmenterType)
      // addNature = false
      .addNature(params.addNature)
      // minTermLen = 1  word.length > 1
      .setMinTermLen(params.minTermLen)
      // minTermNum = 3 一行最少有3个单词
      .setMinTermNum(params.minTermNum)
      .setInputCol("content")
      .setOutputCol("tokens")

    val segDF = segmenter.transform(data)

    val stopwordArray = spark.sparkContext.textFile(params.stopWordFilePath).collect()
    val remover = new StopWordsRemover()
      .setStopWords(stopwordArray)
      .setInputCol(segmenter.getOutputCol)
      .setOutputCol("removed")

    val removedDF = remover.transform(segDF)

    removedDF
  }

  def vectorize(data : DataFrame, params: PreprocessParam) : CountVectorizerModel = {
    val vectorizer = new CountVectorizer()
      .setVocabSize(params.vocabSize)
      .setInputCol("removed")
      .setOutputCol("spark/mllib/features")

    val parentVecModel = vectorizer.fit(data)

    val numPattern = "[0-9]+".r
    val vocabulary = parentVecModel.vocabulary.flatMap {term =>
      if(term.length == 1 || term.matches(numPattern.regex)) None else Some(term)
    }
    val vecModel = new CountVectorizerModel(Identifiable.randomUID("cntVec"),vocabulary)
      .setInputCol("removed")
      .setOutputCol("spark/mllib/features")

    vecModel
  }

  def saveMode(indexModel: StringIndexerModel, vecModel: CountVectorizerModel, params: PreprocessParam): Unit = {
    val indexerModelPath = params.IndexModelPath
    val vecModelPath = params.vecModelPath

    val indexModelFile = new File(indexerModelPath)
    val vecModelFile = new File(vecModelPath)

    if(indexModelFile.exists()) {
      println("索引模型已存在，新模型将覆盖原有模型 ....")
      IOUtils.delDir(indexModelFile)
    }

    indexModel.save(indexerModelPath)
    vecModel.save(vecModelPath)
    println("预处理模型已保存!")
  }

  def train(filePath : String, spark : SparkSession) :
  (DataFrame,StringIndexerModel,CountVectorizerModel) = {
    val parms = new PreprocessParam
    val cleanDF = this.clean(filePath,spark)
    val indexerModel = this.indexrize(cleanDF)
    val indexDF = indexerModel.transform(cleanDF)
    val segDF = this.segment(indexDF,parms)
    val vecModel = this.vectorize(segDF,parms)
    val trainDF = vecModel.transform(segDF)
    this.saveMode(indexerModel,vecModel,parms)

    (trainDF,indexerModel,vecModel)
  }

  def loadModel(params: PreprocessParam) : (StringIndexerModel,CountVectorizerModel) = {
    /*
    * 加载 indexerModel 和  vecModel 这两个模型
    * */
    val indexerModelPath = params.IndexModelPath
    val vecModelPath = params.vecModelPath

    val indexerModelFile = new File(indexerModelPath)
    val vecModelFile = new File(vecModelPath)

    if(! indexerModelFile.exists()) {
      println("索引模型不存在，即将退出")
      System.exit(1)
    } else if(! vecModelFile.exists()) {
      println("向量模型不存在，即将退出")
      System.exit(2)
    } else {
      println("开始加载预处理模型")
    }

    val indexerModel = StringIndexerModel.load(indexerModelPath)
    val vecModel = CountVectorizerModel.load(vecModelPath)
    println("预处模型加载成功！")
    (indexerModel, vecModel)
  }
  def predict(filePath : String, spark : SparkSession) : (DataFrame,StringIndexerModel,CountVectorizerModel) = {
    val params = new PreprocessParam

    val cleanDF = this.clean(filePath,spark)
    val(indexModel,vecModel) = this.loadModel(params)
    // indexDF : org.apache.spark.sql.DataFrame
    val indexDF = indexModel.transform(cleanDF)
    val segDF = this.segment(indexDF,params)
    val predictDF = vecModel.transform(segDF)

    (predictDF , indexModel, vecModel)
  }

}
