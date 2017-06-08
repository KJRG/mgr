package io.github.kjrg.mgr.dto;

import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.Updater;

/**
 * A class representing the information about an experiment.
 * 
 * @author Krzysztof Ga³ka
 */
public class ExperimentInfoDTO {

	private int numberOfNeuronsInHiddenLayer;
	private String activationFunction;
	private Updater updater;
	private Evaluation experimentResult;

	public String getInformationText() {
		StringBuilder messageBuilder = new StringBuilder();
		String lineSeparator = System.lineSeparator();

		messageBuilder.append("Number of neurons in hidden layer: " + numberOfNeuronsInHiddenLayer + lineSeparator);
		messageBuilder.append("Activation function: " + activationFunction + lineSeparator);
		messageBuilder.append("Updater: " + updater + lineSeparator);
		messageBuilder.append(experimentResult.stats());

		return messageBuilder.toString();
	}

	public ExperimentInfoDTO(int numberOfNeuronsInHiddenLayer, String activationFunction, Updater updater,
			Evaluation experimentResult) {
		this.numberOfNeuronsInHiddenLayer = numberOfNeuronsInHiddenLayer;
		this.activationFunction = activationFunction;
		this.updater = updater;
		this.experimentResult = experimentResult;
	}

	public int getNumberOfNeuronsInHiddenLayer() {
		return numberOfNeuronsInHiddenLayer;
	}

	public void setNumberOfNeuronsInHiddenLayer(int numberOfNeuronsInHiddenLayer) {
		this.numberOfNeuronsInHiddenLayer = numberOfNeuronsInHiddenLayer;
	}

	public String getActivationFunction() {
		return activationFunction;
	}

	public void setActivationFunction(String activationFunction) {
		this.activationFunction = activationFunction;
	}

	public Updater getUpdater() {
		return updater;
	}

	public void setUpdater(Updater updater) {
		this.updater = updater;
	}

	public Evaluation getExperimentResult() {
		return experimentResult;
	}

	public void setExperimentResult(Evaluation experimentResult) {
		this.experimentResult = experimentResult;
	}
}
