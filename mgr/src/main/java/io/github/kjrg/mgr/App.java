package io.github.kjrg.mgr;

import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Properties;

import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;

import io.github.kjrg.mgr.dto.ExperimentInfoDTO;

/**
 * Main class of application.
 * 
 * @author Krzysztof Ga³ka
 */
public class App {

	private static final String CONFIGURATION_FILEPATH = "configuration.properties";
	private static final String DEFAULT_FILEPATH_FOR_REPORT = "results.xlsx";
	
	public static void main(String[] args) {
		
		System.out.println("Starting the application...");
		Integer numberOfClasses = null;
		Integer numberOfEpochs = null;
		String outputFilepath = DEFAULT_FILEPATH_FOR_REPORT;

		Properties properties = null;
        DataSet trainDataset = null;
    	DataSet testDataset = null;
    	System.out.println("Loading data...");
    	
		try(InputStream inputStream = App.class.getClassLoader().getResourceAsStream(CONFIGURATION_FILEPATH)) {
			/*
			 * Load configuration.
			 */
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
			outputFilepath = properties.getProperty("report_directory_path");
			
			/*
			 * Load data.
			 */
			DataProvider dataProvider = new DataProvider();
			
			System.out.println("Loading training data from " + trainingDatasetFilepath);
			trainDataset = dataProvider.readDatasetFromFile(trainingDatasetFilepath, trainingDatasetSize, labelColumnIndex, numberOfClasses);
			System.out.println("Loading test data from " + testDatasetFilepath);
			testDataset = dataProvider.readDatasetFromFile(testDatasetFilepath, testDatasetSize, labelColumnIndex, numberOfClasses);

			if (trainDataset == null) {
				System.err.println("The training data could not be read.");
				System.exit(1);
			}
			if (testDataset == null) {
				System.err.println("The test data could not be read.");
				System.exit(1);
			}
		} catch (IOException | InterruptedException e) {
			e.printStackTrace();
			System.exit(1);
		}

		/*
		 * Normalize data.
		 */
		DataNormalization normalizer = new NormalizerStandardize();
		normalizer.fit(trainDataset);
		normalizer.transform(trainDataset);
		normalizer.transform(testDataset);
		
		/*
		 * Get list of neural network configurations for experiments.
		 */
		NeuralNetworkConfigurationProvider neuralNetworkConfigurationProvider = new NeuralNetworkConfigurationProvider();
		List<MultiLayerConfiguration> neuralNetworkConfigurations =
				neuralNetworkConfigurationProvider.readConfigurationsFromProperties(properties);
		
		/*
		 * Run experiments.
		 */
		TestRunner testRunner = new TestRunner();
		ExperimentInfoCreator experimentInfoCreator = new ExperimentInfoCreator();
		List<ExperimentInfoDTO> experimentResultList = new ArrayList<>();

		for (MultiLayerConfiguration configuration : neuralNetworkConfigurations) {	
			Evaluation result = testRunner.runTest(trainDataset, testDataset, configuration, numberOfEpochs, numberOfClasses);
			ExperimentInfoDTO experimentResult = experimentInfoCreator.createInfo(configuration, result);
			experimentResultList.add(experimentResult);
			System.out.println(System.lineSeparator() + experimentResult.getInformationText());
		}
		
		/*
		 * Create report.
		 */
		ReportCreator reportCreator = new ReportCreator();
		try {
			System.out.println(System.lineSeparator() + "Saving report");
			reportCreator.createReport(experimentResultList, outputFilepath);
			System.out.println("Report saved in directory " + outputFilepath);
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
}
