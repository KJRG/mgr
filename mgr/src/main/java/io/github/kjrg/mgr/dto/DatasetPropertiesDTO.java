package io.github.kjrg.mgr.dto;

/**
 * Properties of the dataset.
 * 
 * @author Krzysztof Ga³ka
 */
public class DatasetPropertiesDTO {

	private String trainingDatasetFilepath;
	private String testDatasetFilepath;
	private int trainingDatasetSize;
	private int testDatasetSize;
	private int labelColumnIndex;
	private int numberOfLabels;

	public DatasetPropertiesDTO(String trainingDatasetFilepath, String testDatasetFilepath, int trainingDatasetSize,
			int testDatasetSize, int labelColumnIndex, int numberOfLabels) {
		this.trainingDatasetFilepath = trainingDatasetFilepath;
		this.testDatasetFilepath = testDatasetFilepath;
		this.trainingDatasetSize = trainingDatasetSize;
		this.testDatasetSize = testDatasetSize;
		this.labelColumnIndex = labelColumnIndex;
		this.numberOfLabels = numberOfLabels;
	}

	public String getTrainingDatasetFilepath() {
		return trainingDatasetFilepath;
	}

	public void setTrainingDatasetFilepath(String trainingDatasetFilepath) {
		this.trainingDatasetFilepath = trainingDatasetFilepath;
	}

	public String getTestDatasetFilepath() {
		return testDatasetFilepath;
	}

	public void setTestDatasetFilepath(String testDatasetFilepath) {
		this.testDatasetFilepath = testDatasetFilepath;
	}

	public int getTrainingDatasetSize() {
		return trainingDatasetSize;
	}

	public void setTrainingDatasetSize(int trainingDatasetSize) {
		this.trainingDatasetSize = trainingDatasetSize;
	}

	public int getTestDatasetSize() {
		return testDatasetSize;
	}

	public void setTestDatasetSize(int testDatasetSize) {
		this.testDatasetSize = testDatasetSize;
	}

	public int getLabelColumnIndex() {
		return labelColumnIndex;
	}

	public void setLabelColumnIndex(int labelColumnIndex) {
		this.labelColumnIndex = labelColumnIndex;
	}

	public int getNumberOfLabels() {
		return numberOfLabels;
	}

	public void setNumberOfLabels(int numberOfLabels) {
		this.numberOfLabels = numberOfLabels;
	}

}
