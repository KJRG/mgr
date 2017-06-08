package io.github.kjrg.mgr;

import java.io.IOException;
import java.io.InputStream;
import java.util.Properties;

import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.lossfunctions.LossFunctions;

/**
 * Main class of application.
 */
public class App {

	private static final String CONFIGURATION_FILEPATH = "configuration.properties";
	
	public static void main(String[] args) {
		
		System.out.println("Starting the application...");
		
		int numberOfInputs = 13;
		int numberOfHiddenNeurons = 13;
		int numberOfOutputs = 2;
		
		int iterations = 100;
		int seed = 123;
		int numberOfEpochs = 100;
		double learningRate = 0.05;
		double momentum = 0.9;

		DataProvider dataProvider = new DataProvider();
        DataSet trainDataset = null;
    	DataSet testDataset = null;
    	System.out.println("Reading data...");
    	
		try(InputStream inputStream = App.class.getClassLoader().getResourceAsStream(CONFIGURATION_FILEPATH)) {
			Properties properties = new  Properties();
			if (inputStream != null) {
				properties.load(inputStream);
			} else {
				System.err.println("The configuration file could not be found.");
				System.exit(1);
			}
			
			String trainingDatasetFilepath = properties.getProperty("data.training_dataset_filepath");
			String testDatasetFilepath = properties.getProperty("data.test_dataset_filepath");
			int trainingDatasetSize = Integer.parseInt(properties.getProperty("data.training_dataset_size"));
			int testDatasetSize = Integer.parseInt(properties.getProperty("data.test_dataset_size"));
			int labelColumnIndex = Integer.parseInt(properties.getProperty("data.label_column_index"));
			int numberOfLabels = Integer.parseInt(properties.getProperty("data.number_of_labels"));
			
			System.out.println("Loading training data from " + trainingDatasetFilepath);
			trainDataset = dataProvider.readDatasetFromFile(trainingDatasetFilepath, trainingDatasetSize, labelColumnIndex, numberOfLabels);
			System.out.println("Loading test data from " + testDatasetFilepath);
			testDataset = dataProvider.readDatasetFromFile(testDatasetFilepath, testDatasetSize, labelColumnIndex, numberOfLabels);
			
		} catch (IOException | InterruptedException e) {
			e.printStackTrace();
			System.exit(1);
		}
		
		if (trainDataset == null) {
			System.out.println("The training data could not be read.");
			System.exit(1);
		}
		if (testDataset == null) {
			System.out.println("The test data could not be read.");
			System.exit(1);
		}

		DataNormalization normalizer = new NormalizerStandardize();
		normalizer.fit(trainDataset);
		normalizer.transform(trainDataset);
		normalizer.transform(testDataset);
		
		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
				.seed(seed)
				.iterations(iterations)
				.weightInit(WeightInit.XAVIER)
				.learningRate(learningRate)
				.optimizationAlgo(OptimizationAlgorithm.LINE_GRADIENT_DESCENT)
				.regularization(false)
				.list()
				.layer(0, new DenseLayer.Builder()
						.nIn(numberOfInputs)
						.nOut(numberOfHiddenNeurons)
						.activation("softmax")
						.updater(Updater.ADAM).momentum(momentum)
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
		
		TestRunner testRunner = new TestRunner();
		Evaluation result = testRunner.runTest(trainDataset, testDataset, conf, numberOfEpochs, numberOfOutputs);

		DenseLayer layer = (DenseLayer) conf.getConf(0).getLayer();
        System.out.println("\nNumber of neurons in hidden layer: " + layer.getNIn());
        System.out.println("Activation function: " + layer.getActivationFunction());
        System.out.println("Updater: " + layer.getUpdater());
    	
    	System.out.println(result.stats());
	}
}
