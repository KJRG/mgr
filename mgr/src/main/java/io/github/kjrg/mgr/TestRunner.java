package io.github.kjrg.mgr;

import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;

/**
 * A class for running tests.
 * 
 * @author Krzysztof Ga³ka
 */
public class TestRunner {

	/**
	 * Run a test of a neural network.
	 * 
	 * @param trainDataset training dataset
	 * @param testDataset test dataset
	 * @param configuration the configuration of the neural network to be tested
	 * @param numberOfEpochs number of epochs
	 * @param numberOfClasses number of classes in the dataset
	 * @return
	 */
	public Evaluation runTest(DataSet trainDataset, DataSet testDataset, MultiLayerConfiguration configuration,
			int numberOfEpochs, int numberOfClasses) {
		
		// Create the model
		MultiLayerNetwork model = new MultiLayerNetwork(configuration);
        model.init();
        
        // Train the neural network
    	for (int n = 0; n < numberOfEpochs; n++) {
    		model.fit(trainDataset);
    	}
    	
    	// Perform the test and evaluate the results
    	Evaluation evaluation = new Evaluation(numberOfClasses);

    	INDArray features = testDataset.getFeatureMatrix();
    	INDArray labels = testDataset.getLabels();
    	INDArray predicted = model.output(features, false);
    	
    	evaluation.eval(labels, predicted);
    	return evaluation;
	}
}
