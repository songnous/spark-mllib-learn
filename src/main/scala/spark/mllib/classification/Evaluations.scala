package spark.mllib.classification

import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.rdd.RDD


object Evaluations {
  def multiClassEvaluate(data : RDD[(Double, Double)]) : (Double,Double,Double) = {
    val metrics = new MulticlassMetrics(data)
    val weightedPrecision = metrics.weightedPrecision
    val weightedRecall = metrics.weightedRecall
    val f1 = metrics.weightedFMeasure
    (weightedPrecision,weightedRecall,f1)
  }
}
