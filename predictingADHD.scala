import sqlContext.implicits._
import org.apache.spark.sql._
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.{HashingTF, Tokenizer}
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.sql.Row
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.{HashingTF, Tokenizer}
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.ml.tuning.ParamGridBuilder
import org.apache.spark.ml.tuning.CrossValidator
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator


/** SparkContext variable 'sc' is provided by the driver program running on master node. */
val sqlContext = new org.apache.spark.sql.SQLContext(sc)

/** First, the data is stored in Amazon S3 Bucket. Then it is fetched from there for further processing. */
val trainData = sc.textFile("s3://data/trainingData_Label01.csv")

case class ADHDTrain(vaers_id: String, symptom_text: String, label: Double)

/** After data is loaded it is converted in the form of data frames. */
val adhdtrain = trainData.map(_.split(",")).map(p => ADHDTrain(p(0),p(1),p(2).toDouble))
val adhdtrain_DF = adhdtrain.toDF()

/** Data is in the form of patient's symptom stories, hence to perform text classification, features must be extracted. */
/** This can be done in 2 steps: Tokenizer and Hashed Term Frequency. */

/** Tokenizer breaks the text into bunch of words. So this is appending a new column called “Words” to this DataFrame. */
val tokenizer = new Tokenizer().setInputCol("symptom_text"). setOutputCol("symptom_words")

/** DataFrame is passed to the next module “Hashed Term Freq.” and it outputs a new column called “Features” which is a fixed length vector */
val hashingTF = new HashingTF() .setNumFeatures(1000). setInputColtokenizer.getOutputCol).setOutputCol("features")

/** Training module selects columns 'label' and 'features' and train on those to produce Logistic Regression Model which makes predictions on the new data. */
val lr = new LogisticRegression().setMaxIter(10).setRegParam(0.01)

/** Pipeline is created which is the combination of the three abstractions : Tokenizer, Hashed Term Frequency, LogisticRegression */
/** These 3 can be wrapped into single object for reusability. */
val pipeline = new Pipeline().setStages(Array(tokenizer,hashingTF,lr))

val model = pipeline.fit(adhdtrain_DF)

val predictions = model.transform(adhdtrain_DF)

/** Logistic regression takes the regularization parameter */
/** Hashed term frequency takes various parameters such as length of the representation of feature vector. */
/** CrossValidatior, provided by ML API is used to manage all the different parameters.*/
/** It takes an estimator (the pipeline), Parameter Grid and an Evaluator (to compare models). It then finds the best parameters.  */

val paramGrid = new ParamGridBuilder().addGrid(hashingTF.numFeatures, Array(1000,10000)).addGrid(lr.regParam, Array(0.05,0.2)).build()
val crossval = new CrossValidator().setEstimator(pipeline).setEvaluator(new BinaryClassificationEvaluator).setEstimatorParamMaps(paramGrid).setNumFolds(2)
val cvModel = crossval.fit(adhdtrain_DF)

val evaluator = new BinaryClassificationEvaluator().setMetricName(“areaUnderROC”)
evaluator.evaluate(cvModel.transform(adhdtrain_DF))




