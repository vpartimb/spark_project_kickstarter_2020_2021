package paristech

import org.apache.spark.SparkConf
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.{BinaryClassificationEvaluator, MulticlassClassificationEvaluator}
import org.apache.spark.ml.{Pipeline, PipelineModel, feature => P}

object Trainer {

  Logger.getLogger("org").setLevel(Level.ERROR)

  def main(args: Array[String]): Unit = {

    val conf = new SparkConf().setAll(Map(
      "spark.master"-> "local",
      "spark.scheduler.mode" -> "FIFO",
      "spark.speculation" -> "false",
      "spark.reducer.maxSizeInFlight" -> "48m",
      "spark.serializer" -> "org.apache.spark.serializer.KryoSerializer",
      "spark.kryoserializer.buffer.max" -> "1g",
      "spark.shuffle.file.buffer" -> "32k",
      "spark.default.parallelism" -> "12",
      "spark.sql.shuffle.partitions" -> "12",
      "spark.driver.maxResultSize" -> "2g"
    ))

    val spark = SparkSession
      .builder
      .config(conf)
      .appName("TP Spark : Trainer")
      .getOrCreate()
    import spark.implicits._

    /*******************************************************************************
      *
      *       TP 3
      *
      *       - lire le fichier sauvegarder précédemment
      *       - construire les Stages du pipeline, puis les assembler
      *       - trouver les meilleurs hyperparamètres pour l'entraînement du pipeline avec une grid-search
      *       - Sauvegarder le pipeline entraîné
      *
      *       if problems with unimported modules => sbt plugins update
      *
      ********************************************************************************/

    val df = spark
      .read
      .parquet("/mnt/c/Users/vince/Google Drive/MS BGD/Cours/INF729_Hadoop/spark_project_kickstarter_2020_2021/data/prepared_trainingset/")

    val tokenizer = new P.RegexTokenizer()
      .setPattern("\\W+")
      .setGaps(true)
      .setInputCol("text")
      .setOutputCol("tokens")

    val stopWordsRemover = new P.StopWordsRemover()
      .setInputCol("tokens")
      .setOutputCol("filtered")

    val countVectorizer = new P.CountVectorizer()
      .setMinDF(65)
      .setMinTF(1)
      .setInputCol("filtered")
      .setOutputCol("tf")

    val idf = new P.IDF()
      .setInputCol("tf")
      .setOutputCol("tfidf")

    /** CONVERTING STRINGS TO NUMERICAL VALUES **/

    val countryIndexer = new P.StringIndexer()
      .setInputCol("country2")
      .setOutputCol("country_indexed")
      .setHandleInvalid("keep")

    val currencyIndexer = new P.StringIndexer()
      .setInputCol("currency2")
      .setOutputCol("currency_indexed")
      .setHandleInvalid("keep")

    /** VECTOR ASSEMBLER **/

    val assembler = new P.VectorAssembler()
      .setInputCols(Array("tfidf", "days_campaign", "hours_prepa", "goal", "country_indexed", "currency_indexed"))
      .setOutputCol("features")

    /** LOGISTIC REGRESSION **/

    val lr = new LogisticRegression()
      .setFeaturesCol("features")
      .setLabelCol("final_status")
      .setElasticNetParam(0.0)
      .setFitIntercept(true)
      .setStandardization(true)
      .setPredictionCol("predictions")
      .setRawPredictionCol("raw_predictions")
      .setThresholds(Array(0.7, 0.3))
      .setTol(1.0e-6)
      .setMaxIter(100)

    /** PIPELINE **/

    val pipeline: Pipeline = new Pipeline()
      .setStages(Array(
        tokenizer,
        stopWordsRemover,
        countVectorizer,
        idf,
        countryIndexer,
        currencyIndexer,
        assembler,
        lr
      ))

//    val model = pipeline.fit(df)
//    val dfTransformed = model.transform(df)
//
//    dfTransformed.show()

    df.persist()
    val Array(train, test) = df.randomSplit(Array(0.9, 0.1), seed = 42)
    val model = pipeline.fit(train)

    def evaluateModel(model: PipelineModel, df: DataFrame) = {
      val predictions = model.transform(df)
      val eval_auc = new BinaryClassificationEvaluator()
        .setLabelCol("final_status")
        .setMetricName("areaUnderROC")
        .setRawPredictionCol("probability")

      val eval_wprec = new MulticlassClassificationEvaluator()
        .setMetricName("weightedPrecision")
        .setLabelCol("final_status")
        .setPredictionCol("predictions")

      val eval_recall = new MulticlassClassificationEvaluator()
        .setMetricName("weightedRecall")
        .setLabelCol("final_status")
        .setPredictionCol("predictions")

      val eval_f1 = new MulticlassClassificationEvaluator()
        .setMetricName("f1")
        .setLabelCol("final_status")
        .setPredictionCol("predictions")

      val auc = eval_auc.evaluate(predictions)
      println(s"AUC: $auc")
      val precision = eval_wprec.evaluate(predictions)
      println(s"Precision: $precision")
      val recall = eval_recall.evaluate(predictions)
      println(s"Recall: $recall")
      val f1_score = eval_f1.evaluate(predictions)
      println(s"F1 Score: $f1_score")
    }

    evaluateModel(model, test)

  }
}
