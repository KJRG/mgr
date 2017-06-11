package io.github.kjrg.mgr;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Properties;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.lossfunctions.LossFunctions;

/**
 * Class for generating neural network configurations basing on the parameters
 * from a properties file.
 * 
 * @author Krzysztof Ga³ka
 */
public class NeuralNetworkConfigurationProvider {

	private static final String PARAMETER_SEPARATOR_REGEX = ",";

	/**
	 * Create a list of neural network configurations basing on the properties
	 * from the provided object.
	 * 
	 * @param properties
	 *            an object containing the properties necessary for creating the
	 *            configurations
	 * @return list of neural network configurations
	 */
	public List<MultiLayerConfiguration> readConfigurationsFromProperties(Properties properties) {
		// Get the properties
		int numberOfInputs = Integer.parseInt(properties.getProperty("network_architecture.number_of_inputs"));
		String numbersOfHiddenNeuronsProperty = properties
				.getProperty("network_architecture.numbers_of_hidden_neurons");
		int numberOfClasses = Integer.parseInt(properties.getProperty("network_architecture.number_of_outputs"));

		int seed = Integer.parseInt(properties.getProperty("seed"));
		int iterations = Integer.parseInt(properties.getProperty("iterations"));
		double learningRate = Double.parseDouble(properties.getProperty("learning_rate"));

		String activationFunctionsProperty = properties.getProperty("activation_functions");
		String updatersProperty = properties.getProperty("updaters");

		// Create list of variable parameters of experiments, basing on the
		// values from the properties object
		List<Integer> numbersOfNeuronsInHiddenLayer = getNumbersOfNeuronsInHiddenLayer(numbersOfHiddenNeuronsProperty);
		List<String> activationFunctions = getActivationFunctions(activationFunctionsProperty);
		List<Updater> updaters = getUpdaters(updatersProperty);

		DenseLayer hiddenLayerPrototype = new DenseLayer.Builder()
				.nIn(numberOfInputs)
				.build();

		List<DenseLayer> allPossibleHiddenLayers = new DenseLayerListBuilder(Arrays.asList(hiddenLayerPrototype))
				.withNOuts(numbersOfNeuronsInHiddenLayer)
				.withActivationFunctions(activationFunctions)
				.withUpdaters(updaters)
				.buildList();
		
		List<MultiLayerConfiguration> configurations = new ArrayList<>();
		
		for (DenseLayer hiddenLayer : allPossibleHiddenLayers) {
			int numberOfNeuronsInHiddenLayer = hiddenLayer.getNOut();
			
			MultiLayerConfiguration configuration = new NeuralNetConfiguration.Builder()
					.seed(seed)
					.iterations(iterations)
					.weightInit(WeightInit.XAVIER)
					.learningRate(learningRate)
					.optimizationAlgo(OptimizationAlgorithm.LINE_GRADIENT_DESCENT)
					.regularization(false)
					.list()
					.layer(0, hiddenLayer)
					.layer(1, new OutputLayer.Builder(
							LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
							.activation("softmax")
							.nIn(numberOfNeuronsInHiddenLayer)
							.nOut(numberOfClasses)
							.build())
					.backprop(true)
					.pretrain(false)
					.build();
			
			configurations.add(configuration);
		}
		
		return configurations;
	}

	private List<Integer> getNumbersOfNeuronsInHiddenLayer(String numbersOfHiddenNeuronsProperty) {
		if (numbersOfHiddenNeuronsProperty == null) {
			throw new IllegalStateException("The number of neurons in hidden layer was not provided.");
		}

		List<Integer> numbersOfNeuronsInHiddenLayer = new ArrayList<>();
		String[] parameters = numbersOfHiddenNeuronsProperty.split(PARAMETER_SEPARATOR_REGEX);

		if (parameters.length == 0) {
			throw new IllegalStateException("The number of neurons in hidden layer was not provided or is incorrect.");
		}

		for (String parameter : parameters) {
			if (parameter == null || parameter.isEmpty()) {
				System.out.println("Warning: the number of neurons in hidden layer " + parameter
						+ " is not correct and will be ignored");
				continue;
			}

			Integer numberOfNeuronsInHiddenLayer = null;
			try {
				numberOfNeuronsInHiddenLayer = Integer.parseInt(parameter);
			} catch (NumberFormatException e) {
				System.out.println("Warning: the number of neurons in hidden layer " + parameter
						+ " is not correct and will be ignored");
				continue;
			}

			if (numberOfNeuronsInHiddenLayer < 1) {
				System.out.println("Warning: the number of neurons in hidden layer " + parameter
						+ " is not correct and will be ignored");
				System.out.println("The number of neutrons in hidden layer has to be at least 1.");
				continue;
			}

			numbersOfNeuronsInHiddenLayer.add(numberOfNeuronsInHiddenLayer);
		}

		return numbersOfNeuronsInHiddenLayer;
	}

	private List<String> getActivationFunctions(String activationFunctionsProperty) {
		if (activationFunctionsProperty == null) {
			throw new IllegalStateException("No activation functions were provided.");
		}

		List<String> activationFunctions = new ArrayList<>();
		String[] parameters = activationFunctionsProperty.split(PARAMETER_SEPARATOR_REGEX);

		if (parameters.length == 0) {
			throw new IllegalStateException("The activation functions were not provided or are incorrect.");
		}

		List<String> allowedActivationFunctions = Arrays.asList("relu", "tanh", "sigmoid", "softmax", "hardtanh",
				"leakyrelu", "maxout", "softsign", "softplus");

		for (String parameter : parameters) {
			if (parameter == null || parameter.isEmpty() || !allowedActivationFunctions.contains(parameter)) {
				System.out.println(
						"Warning: the activation function " + parameter + " is not correct and will be ignored");
				continue;
			}

			activationFunctions.add(parameter);
		}

		return activationFunctions;
	}

	private List<Updater> getUpdaters(String updatersProperty) {
		if (updatersProperty == null) {
			throw new IllegalStateException("No updaters were provided.");
		}

		List<Updater> updaters = new ArrayList<>();
		String[] parameters = updatersProperty.split(PARAMETER_SEPARATOR_REGEX);

		if (parameters.length == 0) {
			throw new IllegalStateException("The updaters were not provided or are incorrect.");
		}

		for (String parameter : parameters) {
			if (parameter == null || parameter.isEmpty()) {
				System.out.println("Warning: the updater " + parameter + " is not correct and will be ignored");
				continue;
			}

			try {
				updaters.add(Updater.valueOf(parameter.toUpperCase()));
			} catch (IllegalArgumentException e) {
				System.out.println("Warning: the updater " + parameter + " is not correct and will be ignored");
				continue;
			}
		}

		return updaters;
	}

	private static class DenseLayerListBuilder {

		private List<DenseLayer> layers = new ArrayList<>();

		public DenseLayerListBuilder(List<DenseLayer> layers) {
			this.layers = layers;
		}

		public DenseLayerListBuilder withNOuts(List<Integer> nOuts) {
			List<DenseLayer> newLayers = new ArrayList<>();

			for (DenseLayer layer : this.layers) {
				for (Integer nOut : nOuts) {
					DenseLayer newLayer = (DenseLayer) layer.clone();
					newLayer.setNOut(nOut);
					newLayers.add(newLayer);
				}
			}

			this.layers = newLayers;
			return this;
		}

		public DenseLayerListBuilder withActivationFunctions(List<String> activationFunctions) {
			List<DenseLayer> newLayers = new ArrayList<>();

			for (DenseLayer layer : this.layers) {
				for (String activationFunction : activationFunctions) {
					DenseLayer newLayer = (DenseLayer) layer.clone();
					newLayer.setActivationFunction(activationFunction);
					newLayers.add(newLayer);
				}
			}

			this.layers = newLayers;
			return this;
		}

		public DenseLayerListBuilder withUpdaters(List<Updater> updaters) {
			List<DenseLayer> newLayers = new ArrayList<>();

			for (DenseLayer layer : this.layers) {
				for (Updater updater : updaters) {
					DenseLayer newLayer = (DenseLayer) layer.clone();
					newLayer.setUpdater(updater);
					newLayers.add(newLayer);
				}
			}

			this.layers = newLayers;
			return this;
		}

		public List<DenseLayer> buildList() {
			return this.layers;
		}
	}
}
