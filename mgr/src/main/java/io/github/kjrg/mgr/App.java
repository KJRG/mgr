package io.github.kjrg.mgr;

import java.io.IOException;
import java.io.InputStream;
import java.util.List;
import java.util.Properties;

import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;

/**
 * Main class of application.
 */
public class App {

	private static final String CONFIGURATION_FILEPATH = "configuration.properties";
	
	public static void main(String[] args) {
		
		System.out.println("Starting the application...");
		Integer numberOfClasses = null;
		Integer numberOfEpochs = null;

		Properties properties = null;
        DataSet trainDataset = null;
    	DataSet testDataset = null;
    	System.out.println("Loading data...");
    	
		try(InputStream inputStream = App.class.getClassLoader().getResourceAsStream(CONFIGURATION_FILEPATH)) {
			properties = new Properties();
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
			numberOfClasses = Integer.parseInt(properties.getProperty("data.number_of_labels"));
			numberOfEpochs = Integer.parseInt(properties.getProperty("number_of_epochs"));
			
			DataProvider dataProvider = new DataProvider();
			
			System.out.println("Loading training data from " + trainingDatasetFilepath);
			trainDataset = dataProvider.readDatasetFromFile(trainingDatasetFilepath, trainingDatasetSize, labelColumnIndex, numberOfClasses);
			System.out.println("Loading test data from " + testDatasetFilepath);
			testDataset = dataProvider.readDatasetFromFile(testDatasetFilepath, testDatasetSize, labelColumnIndex, numberOfClasses);
			
		} catch (IOException | InterruptedException e) {
			e.printStackTrace();
			System.exit(1);
		}
		
		if (trainDataset == null) {
			System.err.println("The training data could not be read.");
			System.exit(1);
		}
		if (testDataset == null) {
			System.err.println("The test data could not be read.");
			System.exit(1);
		}

		DataNormalization normalizer = new NormalizerStandardize();
		normalizer.fit(trainDataset);
		normalizer.transform(trainDataset);
		normalizer.transform(testDataset);
		
		NeuralNetworkConfigurationProvider neuralNetworkConfigurationProvider = new NeuralNetworkConfigurationProvider();
		List<MultiLayerConfiguration> neuralNetworkConfigurations =
				neuralNetworkConfigurationProvider.readConfigurationsFromProperties(properties);
		
		TestRunner testRunner = new TestRunner();
		ExperimentInfoCreator experimentInfoCreator = new ExperimentInfoCreator();
		
		for (MultiLayerConfiguration configuration : neuralNetworkConfigurations) {	
			Evaluation result = testRunner.runTest(trainDataset, testDataset, configuration, numberOfEpochs, numberOfClasses);
			System.out.println(experimentInfoCreator.createInfo(configuration, result).getInformationText());
		}
	}
}
