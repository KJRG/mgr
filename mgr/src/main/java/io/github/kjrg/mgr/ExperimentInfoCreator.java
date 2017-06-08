package io.github.kjrg.mgr;

import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;

import io.github.kjrg.mgr.dto.ExperimentInfoDTO;

/**
 * A class for creating the information about an experiment.
 * 
 * @author Krzysztof Ga³ka
 */
public class ExperimentInfoCreator {

	private static final int INDEX_OF_HIDDEN_LAYER = 0;

	/**
	 * Create information about an experiment.
	 * 
	 * @param neuralNetworkConfiguration configuration about the neural network
	 * @param experimentResult result of the experiment
	 * @return information about experiment
	 */
	public ExperimentInfoDTO createInfo(MultiLayerConfiguration neuralNetworkConfiguration,
			Evaluation experimentResult) {

		NeuralNetConfiguration configuration = neuralNetworkConfiguration.getConf(INDEX_OF_HIDDEN_LAYER);
		if (configuration == null) {
			throw new IllegalStateException("The network has no hidden layer");
		}
		
		DenseLayer layer = (DenseLayer) configuration.getLayer();
		if (layer == null) {
			throw new IllegalStateException("The network has no hidden layer");
		}

		return new ExperimentInfoDTO(layer.getNIn(), layer.getActivationFunction(), layer.getUpdater(), experimentResult);
	}
}
