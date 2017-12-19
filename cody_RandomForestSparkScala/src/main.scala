import java.io.PrintWriter
import org.apache.log4j._
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.RFormula
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.types._
import org.apache.spark.sql.functions._

object main {
  def main(args: Array[String]) {
    //args(0) is path where you want to save mushroom model
    //args(1) is the path you want to save the confusion matrix
    //args(2) is the location of the mushroom.csv
    println("These are the args! " + args.toList)

    val log = LogManager.getRootLogger
    val conf = new SparkConf().setAppName("CS596FinalProject").setMaster("local[*]").set("spark.sql.session.timeZone", "PST")

    val sc = new SparkContext(conf)
    val spark = SparkSession.builder().config(conf).getOrCreate()

    log.info("Start")
    def ConfusionMatrix(predictionData: DataFrame, logger : Logger, saveToFile : String , filename : String) : String = {
      val truePostive = predictionData.filter("label == 1.0").filter("prediction == 1.0").count
      val falseNegative = predictionData.filter("label == 1.0").filter("prediction == 0.0").count

      val falsePositive = predictionData.filter("label == 0.0").filter("prediction == 1.0").count
      val trueNegative = predictionData.filter("label == 0.0").filter("prediction == 0.0").count

      var log = ""
      var statistics = ""
      log+="Confusion Matrix\n[TP FP]\n[FN TN]\n"
      log+=truePostive + "\t" + falsePositive + "\n"
      log+=falseNegative + "\t" + trueNegative + "\n"

      val accuracy = (truePostive + trueNegative) / (falsePositive + truePostive + falseNegative + trueNegative).toDouble
      val errorRate = (falsePositive + falseNegative) / (falsePositive + truePostive + falseNegative + trueNegative).toDouble
      log += s"Accuracy: $accuracy\n"
      log += s"Error Rate: $errorRate\n"
      statistics = s"$accuracy, $errorRate"
      if (falsePositive + truePostive != 0) {
        val percentFalsePos = falsePositive / (falsePositive + truePostive).toDouble
        log += s"Percent false positive: $percentFalsePos\n"
        statistics += s", $percentFalsePos"
      }
      else {
        log += "No Predicted true cases\n"
        statistics += ", nil"
      }
      if (falseNegative + trueNegative != 0) {
        val percentFalseNegative = falseNegative / (falseNegative + trueNegative).toDouble
        log += s"Percent false negative: $percentFalseNegative\n"
        statistics += s", $percentFalseNegative\n"
      }
      else {
        log += "No Predicted false cases\n"
        statistics += ", nil\n"
      }

      logger.info(log)
      //create an RDD of the log, only thing that seems to work for both AWS and my local machine for saving the output of the confusion matrix on AWS
      //sc.parallelize(List(log)).coalesce(1).saveAsTextFile(saveToFile + "/" + filename)

      val fileWriter = new PrintWriter(saveToFile + "/" + filename)
      fileWriter.write(log)
      fileWriter.close()

      statistics
    }


    val variableList = Array("cap_shape"
      , "cap_surface"
      , "cap_color"
      , "bruises"
      , "odor"
      , "gill_attachment"
      , "gill_spacing"
      , "gill_size"
      , "gill_color"
      , "stalk_shape"
      , "stalk_root"
      , "stalk_surface_above_ring"
      , "stalk_surface_below_ring"
      , "stalk_color_above_ring"
      , "stalk_color_below_ring"
      , "veil_color"
      , "ring_number"
      , "ring_type"
      , "spore"
      , "population"
      , "habitat")

    val schema = new StructType(Array(
      new StructField("classification", StringType),
      new StructField("cap_shape", StringType),
      new StructField("cap_surface", StringType),
      new StructField("cap_color", StringType),
      new StructField("bruises", StringType),
      new StructField("odor", StringType),
      new StructField("gill_attachment", StringType),
      new StructField("gill_spacing", StringType),
      new StructField("gill_size", StringType),
      new StructField("gill_color", StringType),
      new StructField("stalk_shape", StringType),
      new StructField("stalk_root", StringType),
      new StructField("stalk_surface_above_ring", StringType),
      new StructField("stalk_surface_below_ring", StringType),
      new StructField("stalk_color_above_ring", StringType),
      new StructField("stalk_color_below_ring", StringType),
      new StructField("veil_type", StringType),
      new StructField("veil_color", StringType),
      new StructField("ring_number", StringType),
      new StructField("ring_type", StringType),
      new StructField("spore", StringType),
      new StructField("population", StringType),
      new StructField("habitat", StringType)))

    val reader = spark.read
    reader.option("delimiter", ",")
    reader.option("header", false)
    val mushroomFilename = args(2)

    var data = reader.schema(schema).csv(mushroomFilename)
    //veil_type never varies in this data set, throwing out as irrelevant
    data = data.drop(col("veil_type"))

    def classifyEachVariable(variableArray : Array[String]) : String = {
      var csvContent = "Variable, Accuracy, ErrorRate, PercentFalsePositive, PercentFalseNegative\n"
      val t0 = System.nanoTime()
      for(i <- 0 to variableArray.length - 1) {
        //Create the label and feature vector
        val supervised = new RFormula().setFormula("classification ~ " + variableArray(i))
        val fitted = supervised.fit(data)
        val preparedDF = fitted.transform(data)
        //split the data 70/.30
        val Array(train, test) = preparedDF.randomSplit(Array(0.7, 0.3))
        //Setup the random forest classifier
        val dt = new RandomForestClassifier().setLabelCol("label").setFeaturesCol("features")
        //Create a pipeline for cross validation
        val pipeline = new Pipeline().setStages(Array(dt))
        val paramGrid = new ParamGridBuilder().build()

        val evaluator = new MulticlassClassificationEvaluator()
          .setLabelCol("label")
          .setPredictionCol("prediction")
        //Set number of folds to 10
        val cv = new CrossValidator()
          .setEstimator(pipeline)
          .setEvaluator(evaluator)
          .setEstimatorParamMaps(paramGrid)
          .setNumFolds(10)
        //Generate the model
        val model = cv.fit(train)
        //Apply the model to the testing set
        val prediction = model.transform(test)

        csvContent += s"${variableArray(i)}, ${ConfusionMatrix(prediction, log, args(1), variableArray(i) + "ConfusionMatrix.txt")}"
      }
      val t1 = System.nanoTime()
      val timeExpired = t1 - t0
      println(s"Total algorithm time ${timeExpired/1e9}")
      csvContent
    }

    def classifyEverything(variableArray : Array[String]) : String = {
      var csvContent = "Variable, Accuracy, ErrorRate, PercentFalsePositive, PercentFalseNegative\n"
      variableArray.reduceLeft(_ + " + " + _)
      val t0 = System.nanoTime()
      //Create the label and feature vector
      val supervised = new RFormula().setFormula("classification ~ " + variableArray.reduceLeft(_ + " + " + _))
      val fitted = supervised.fit(data)
      val preparedDF = fitted.transform(data)
      //split the data 70/.30
      val Array(train, test) = preparedDF.randomSplit(Array(0.7, 0.3))
      //Setup the random forest classifier
      val dt = new RandomForestClassifier().setLabelCol("label").setFeaturesCol("features")
      //Create a pipeline for cross validation
      val pipeline = new Pipeline().setStages(Array(dt))
      val paramGrid = new ParamGridBuilder().build()

      val evaluator = new MulticlassClassificationEvaluator()
        .setLabelCol("label")
        .setPredictionCol("prediction")
      //Set number of folds to 10
      val cv = new CrossValidator()
        .setEstimator(pipeline)
        .setEvaluator(evaluator)
        .setEstimatorParamMaps(paramGrid)
        .setNumFolds(10)
      //Generate the model
      val model = cv.fit(train)
      //Apply the model to the testing set
      val prediction = model.transform(test)
      val t1 = System.nanoTime()
      val timeExpired = t1 - t0
      println(s"Total algorithm time ${timeExpired/1e9}")
      csvContent += s"[${variableArray.reduceLeft(_ + "," + _)}], ${ConfusionMatrix(prediction, log, args(1), "AllVariables" + "ConfusionMatrix.txt")}"
      csvContent
    }

    def classifyTopTwoFeatures() : String = {
      var csvContent = "Variable, Accuracy, ErrorRate, PercentFalsePositive, PercentFalseNegative\n"
      val t0 = System.nanoTime()
      //Create the label and feature vector
      val supervised = new RFormula().setFormula("classification ~ spore + odor")
      val fitted = supervised.fit(data)
      val preparedDF = fitted.transform(data)
      //split the data 70/.30
      val Array(train, test) = preparedDF.randomSplit(Array(0.7, 0.3))
      //Setup the random forest classifier
      val dt = new RandomForestClassifier().setLabelCol("label").setFeaturesCol("features")
      //Create a pipeline for cross validation
      val pipeline = new Pipeline().setStages(Array(dt))
      val paramGrid = new ParamGridBuilder().build()

      val evaluator = new MulticlassClassificationEvaluator()
        .setLabelCol("label")
        .setPredictionCol("prediction")
      //Set number of folds to 10
      val cv = new CrossValidator()
        .setEstimator(pipeline)
        .setEvaluator(evaluator)
        .setEstimatorParamMaps(paramGrid)
        .setNumFolds(10)
      //Generate the model
      val model = cv.fit(train)
      //Apply the model to the testing set
      val prediction = model.transform(test)
      val t1 = System.nanoTime()
      val timeExpired = t1 - t0
      println(s"Total algorithm time ${timeExpired/1e9}")
      csvContent += s"[spore, odor], ${ConfusionMatrix(prediction, log, args(1), "TopTwoVariables" + "ConfusionMatrix.txt")}"
      csvContent
    }

    val summaryWriter = new PrintWriter("SummaryOfResults.csv")
    summaryWriter.write(classifyEachVariable(variableList))
    summaryWriter.close()

    val summaryWriterAll = new PrintWriter("SummaryOfAllVariablesResults.csv")
    summaryWriterAll.write(classifyEverything(variableList))
    summaryWriterAll.close()

    val summaryWriterTopTwo = new PrintWriter("SummaryOfTopTwoVariablesResults.csv")
    summaryWriterTopTwo.write(classifyTopTwoFeatures())
    summaryWriterTopTwo.close()
  }
}