from numpy import array
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.tree import DecisionTree, DecisionTreeModel
from pyspark import SparkContext
from pyspark.mllib.evaluation import BinaryClassificationMetrics

sc = SparkContext(appName="PythonDecisionTreeClassificationExample")
data = [
     LabeledPoint(0.0, [0.0]),
     LabeledPoint(1.0, [1.0]),
     LabeledPoint(0.0, [-2.0]),
     LabeledPoint(0.0, [-1.0]),
     LabeledPoint(0.0, [-3.0]),
     LabeledPoint(1.0, [4.0]),
     LabeledPoint(1.0, [4.5]),
     LabeledPoint(1.0, [4.9]),
     LabeledPoint(1.0, [3.0])
 ]
all_data = sc.parallelize(data)
(trainingData, testData) = all_data.randomSplit([0.8, 0.2])

# model = DecisionTree.trainClassifier(sc.parallelize(data), 2, {})
model = DecisionTree.trainClassifier(trainingData, numClasses=2, categoricalFeaturesInfo={},
                                         impurity='gini', maxDepth=5, maxBins=32)
print(model)
print(model.toDebugString())
model.predict(array([1.0]))
model.predict(array([0.0]))
rdd = sc.parallelize([[1.0], [0.0]])
model.predict(rdd).collect()

predictions = model.predict(testData.map(lambda x: x.features))
labelsAndPredictions = testData.map(lambda lp: lp.label).zip(predictions)
predictionsAndLabels = predictions.zip(testData.map(lambda lp: lp.label))

metrics = BinaryClassificationMetrics(predictionsAndLabels )

testErr = labelsAndPredictions.filter(lambda v_p: v_p[0] != v_p[1]).count() / float(testData.count())
print('Test Error = ' + str(testErr))
print('Learned classification tree model:')
print(model.toDebugString())

# Save and load model
model.save(sc, "./myDecisionTreeClassificationModel")
sameModel = DecisionTreeModel.load(sc, "./myDecisionTreeClassificationModel")