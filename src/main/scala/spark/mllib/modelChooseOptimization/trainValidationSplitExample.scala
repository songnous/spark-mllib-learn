package spark.mllib.modelChooseOptimization

import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.sql.SparkSession

object trainValidationSplitExample {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder()
      .master("local")
      .appName("trainValidationSplitExample")
      .config("spark.testing.memory", "471859200")
      .getOrCreate()

    val input = args(0)
    val data = spark.read.format("libsvm").load(input)
    val Array(training,test) = data.randomSplit(Array(0.9,0.1),seed = 12345)

    val lr = new LinearRegression()
      .setMaxIter(10)

    // 我们使用 ParamGridBuilder 构建一个搜索参数网格
    // TranValidationSplit 将尝试所有的组合，并确定最佳模型
    val paramGrid = new ParamGridBuilder()
      .addGrid(lr.regParam,Array(0.1,0.01))
      .addGrid(lr.fitIntercept)
      .addGrid(lr.elasticNetParam,Array(0.0,0.5,1.0))
      .build()

    // 在这种情况下，估计量就是线性回归
    // TranValidationSplit 需要估计器，一组估计器参数映射和一个评估器
    val trainValidationSplit = new TrainValidationSplit()
      .setEstimator(lr)
      .setEvaluator(new RegressionEvaluator)
      .setEstimatorParamMaps(paramGrid)
      .setTrainRatio(0.8)
    // 80%  用于培训，20% 用于验证

    // 使用验证拆分训练，并选择最佳的一组参数组合
    val model = trainValidationSplit.fit(training)

    // 对测试数据进行预测，模型具有最佳性能的参数组合

    model.transform(test)
      .select("spark/mllib/features","label","prediction")
      .show(false)

  }
}
