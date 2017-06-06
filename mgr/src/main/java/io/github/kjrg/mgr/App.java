package io.github.kjrg.mgr;

import java.io.IOException;

import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.lossfunctions.LossFunctions;

/**
 * Main class of application.
 */
public class App {

	private static final String TRAINING_DATASET_FILEPATH = "D:\\Praca magisterska\\dane\\cleveland_binary_training_data.csv";
	private static final String TEST_DATASET_FILEPATH = "D:\\Praca magisterska\\dane\\cleveland_binary_test_data.csv";

	public static void main(String[] args) {
		
		System.out.println("Starting the application...");
		
		int numberOfInputs = 13;
		int numberOfHiddenNeurons = 2;
		int numberOfOutputs = 2;
		
		int trainingDatasetSize = 238;
		int testDatasetSize = 59;
		int labelColumnIndex = 13;
		int numberOfLabels = 2;
		
		int iterations = 100;
		int seed = 123;
		int numberOfEpochs = 100;
		double learningRate = 0.05;
		double momentum = 0.9;

		DataProvider dataProvider = new DataProvider();
        DataSet trainDataset = null;
    	DataSet testDataset = null;
    	System.out.println("Reading data...");
		try {
			trainDataset = dataProvider.readDatasetFromFile(TRAINING_DATASET_FILEPATH, trainingDatasetSize, labelColumnIndex, numberOfLabels);
			testDataset = dataProvider.readDatasetFromFile(TEST_DATASET_FILEPATH, testDatasetSize, labelColumnIndex, numberOfLabels);
			
			DataNormalization normalizer = new NormalizerStandardize();
			normalizer.fit(trainDataset);
			normalizer.transform(trainDataset);
			normalizer.transform(testDataset);
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
		
		MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        
        UIServer uiServer = UIServer.getInstance();
        StatsStorage statsStorage = new InMemoryStatsStorage();         //Alternative: new FileStatsStorage(File), for saving and loading later
        uiServer.attach(statsStorage);
        model.setListeners(new StatsListener(statsStorage));
        
    	for (int n = 0; n < numberOfEpochs; n++) {
    		model.fit(trainDataset);
    	}
    	
    	Evaluation eval = new Evaluation(numberOfOutputs);

    	INDArray features = testDataset.getFeatureMatrix();
    	INDArray labels = testDataset.getLabels();
    	INDArray predicted = model.output(features, false);
    	
    	eval.eval(labels, predicted);
    	
    	System.out.println(eval.stats());
	}
}
