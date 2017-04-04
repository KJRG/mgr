package io.github.kjrg.mgr;

import java.io.File;
import java.io.IOException;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.KFoldIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions;

/**
 * Main class of application.
 *
 */
public class App {

	private static final String DATASET_FILEPATH = "D:\\Praca magisterska\\dane\\cleveland_binary.csv";
	private static final String COMMA_SEPARATOR = ";";

	public static void main(String[] args) {
		
		System.out.println("Hello World!");
		
		int numberOfInputs = 13;
		int numberOfHiddenNeurons = 5;
		int numberOfOutputs = 2;
		
		int numberOfSamples = 297;
		int batchSize = numberOfSamples;
		int iterations = 100;
		int seed = 123;
		int numberOfEpochs = 30;
		double learningRate = 0.05;
		
		int labelColumnIndex = 13;
		int numberOfLabels = 2;
		
		int numberOfFolds = 10;

		RecordReader recordReader = new CSVRecordReader(0, COMMA_SEPARATOR);
		try {
			recordReader.initialize(new FileSplit(new File(DATASET_FILEPATH)));
		} catch (IOException | InterruptedException e) {
			e.printStackTrace();
			System.exit(1);
		}
		DataSetIterator datasetIterator = new RecordReaderDataSetIterator(recordReader, batchSize, labelColumnIndex, numberOfLabels);
		DataSet dataset = datasetIterator.next();

		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
				.seed(seed)
				.iterations(iterations)
				.weightInit(WeightInit.XAVIER)
				.learningRate(learningRate)
				.optimizationAlgo(OptimizationAlgorithm.LINE_GRADIENT_DESCENT)
				.regularization(true).l2(1e-4)
				.list()
				.layer(0, new DenseLayer.Builder()
						.nIn(numberOfInputs)
						.nOut(numberOfHiddenNeurons)
						.activation("tanh")
						.updater(Updater.ADADELTA)
						.build())
				.layer(1, new OutputLayer.Builder(
						LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
						.activation("softmax")
						.nIn(numberOfHiddenNeurons)
						.nOut(numberOfOutputs)
						.build())
				.backprop(true)
				.pretrain(false)
				.build();
		
		MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        
        KFoldIterator kFoldIterator = new KFoldIterator(numberOfFolds, dataset);
        kFoldIterator.reset();
        while (kFoldIterator.hasNext()) {
        	DataSet trainDataset = kFoldIterator.next();
        	DataSet testDataset = kFoldIterator.testFold();

        	for (int n = 0; n < numberOfEpochs; n++) {
        		model.fit(trainDataset);
        	}
        	
        	System.out.println("Fold " + kFoldIterator.cursor() + ", evaluate model...");
        	Evaluation eval = new Evaluation(numberOfOutputs);

        	INDArray features = testDataset.getFeatureMatrix();
        	INDArray labels = testDataset.getLabels();
        	INDArray predicted = model.output(features, false);
        	
        	eval.eval(labels, predicted);
        	
        	System.out.println(eval.stats());
        }
	}
}
