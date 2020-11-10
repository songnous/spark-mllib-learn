package spark.mllib.classification

import com.hankcs.hanlp.seg.Segment
import com.hankcs.hanlp.seg.common.Term
import com.hankcs.hanlp.tokenizer.{IndexTokenizer, NLPTokenizer, SpeedTokenizer, StandardTokenizer}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.{DataFrame, Row}

// 将 java 集合转换成 scala
import scala.collection.JavaConversions._

class Segmenter(val uid : String) extends  Serializable {
  //(val uid : String) 主构造器
  // 英文字符正则
  private val enExpr = "[a-zA-Z]+"
  // 数值正则
  private val numExpr = "\\d+(\\d.\\d+)?(\\/\\d+)"
  // 匹配英文字母，数字，中文汉字之外的字符
  private val baseExpr =
    """[^\w-\s+\u4e00-\u9fa5]"""

  private var inputCol: String = ""
  private var outputCol: String = ""
  private var segmentType: String = "StandardTokenizer"
  private var addNature: Boolean = false
  private var delNum: Boolean = false
  private var delEn: Boolean = false
  private var minTermLen: Int = 1
  private var minTermNum: Int = 3

  /*
  * class A {
  * def method1 : this.type = this
  * }
  * class B extends A {
  * def method2 : this.type = this
  * }
  * val b  = new B
  * b.method1.method2  这个不报错
  * b.method2.method1  这个同样不报错
  * 返回 this 就可以实现，链式调用，如: b.method1.method2
  * 使用 this.type 子类对象调用父类方法的时候返回的是子类类型，而不是父类的类型
  * */
  def setInputCol(value: String): this.type = {
    this.inputCol = value
    this
  }

  def setOutputCol(value: String): this.type = {
    this.outputCol = value
    this
  }

  def setSegmentType(value: String): this.type = {
    this.segmentType = value
    this
  }

  def addNature(value: Boolean): this.type = {
    this.addNature = value
    this
  }

  def isDelNum(value: Boolean): this.type = {
    this.delNum = value
    this
  }

  def isDelEn(value: Boolean): this.type = {
    this.delEn = value
    this
  }

  def setMinTermLen(value: Int): this.type = {
    require(value > 0, "行最小长度必须大于0")
    this.minTermLen = value
    this
  }

  def setMinTermNum(value: Int): this.type = {
    // require 对函数的参数值进行限制，不满足条件将抛异常
    require(value > 0, "行最小词数必须大于0")
    this.minTermNum = value
    this
  }

  def getInputCol: String = this.inputCol

  def getOutputCol: String = this.outputCol

  def getSegmentType: String = this.segmentType


  /*
  * Identifiable.randomUID 生成一个随机字符串
  * String = segment_4f58d56be49f
  * def this() 这是一个辅助构造器
  * this(Identifiable.randomUID("segment")) 调用主构造器
  * 辅助构造器必须以一个对主构造器或其他已存在的辅助构造器的调用开始
  * */
  def this() = this(Identifiable.randomUID("segment"))

  def transform(dataset: DataFrame): DataFrame = {
    // spark : org.apache.spark.sql.SparkSession
    val spark = dataset.sparkSession
    var segment: Segment = null

    segmentType match {
        // NShortSegment,CRFSegment 这两个类，没有序列化，直接在 RDD 中使用会报错,所以自定义两个类，
      // 分别继承它们和 Serializable,这两个自定义的类是空的，啥也不做，但是已经达到了目的，既可以当
        // 原有的类使用，同时还序列化
      case "NShortSegment" => segment = new MyNShortSegment()
      case "CRFSegment" => segment = new MyCRFSegment()
      case _ =>
    }

    val tokens = dataset.select(inputCol).rdd.flatMap { case Row(line: String) =>
      var terms: Seq[Term] = Seq()
      /*
      * StandardTokenizer : 标准分词器
      * NLPSegment NLP 场景的分词器，更注重准确性
      * IndexSegment 索引分词器
      * SpeedTokenizer 极速分词器，适用于高吞吐，不注重准确性
      * NShortSegment 实词分词器，自动移除停用词
      * CRFSegment CRF词法分析器（中文分词、词性标注和命名实体识别） 已经废弃，
      * 请使用功能更丰富、设计更优雅的CRFLexicalAnalyzer
      * segment(line : String) : List[Term]  java.util.List
      * seg(line:String) : List[Term] java.util.List
      * seg 是 Segment 类的方法
      * MyNShortSegment extends NShortSegment extends WordBasedSegment extends Segment
      * MyCRFSegment extends CRFSegment extends CharacterBasedSegment extends Segment
      * */
      segmentType match {
        case "StandardSegment" => terms = StandardTokenizer.segment(line)
        case "NLPSegment" => terms = NLPTokenizer.segment(line)
        case "IndexSegment" => terms = IndexTokenizer.segment(line)
        case "SpeedTokenizer" => terms = SpeedTokenizer.segment(line)
        case "NShortSegment" => terms = segment.seg(line)
        case "CRFSegment" => terms = segment.seg(line)
        case _ => println("分词类型错误")
          System.exit(1)
      }

        val termSeq = terms.flatMap { term =>
          // term ： Term 类的对象，有 word ，offset和 nature 三个成员
          val word = term.word
          val nature = term.nature

          // word : String
          // string.matches(regex:abc)  相当于 ^abc$ java 会自动加上 ^ $。返回 true 或 false。表示匹配是否成功
          // 过滤掉数字、英文，字符串长度小于3
          if(this.delNum && word.matches(numExpr)) None
          else if(this.delEn && word.matches(enExpr)) None
          else if(word.length < minTermNum) None
          else if(this.addNature) Some(word + "/" + nature)
          else Some(word)
        }
        // 过滤词数较少的行
        if(termSeq.nonEmpty && termSeq.size >= minTermNum) Some(line,termSeq) else None
    }

    import spark.implicits._
    val tokenSet = tokens.toDF(inputCol + "#1",outputCol)
    dataset.join(tokenSet,dataset(inputCol) === tokenSet(inputCol + "#1")).
      drop(inputCol + "#1")

  }
}
