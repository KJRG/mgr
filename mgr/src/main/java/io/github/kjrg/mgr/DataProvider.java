package io.github.kjrg.mgr;

import java.io.File;
import java.io.IOException;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

public class DataProvider {

	private static final int LINES_TO_SKIP = 0;
	private static final String SEPARATOR = ";";

	public DataSet readDatasetFromFile(String filepath, int batchSize, int labelColumnIndex, int numberOfLabels)
			throws IOException, InterruptedException {
		RecordReader recordReader = new CSVRecordReader(LINES_TO_SKIP, SEPARATOR);
		recordReader.initialize(new FileSplit(new File(filepath)));
		DataSetIterator datasetIterator = new RecordReaderDataSetIterator(recordReader, batchSize, labelColumnIndex,
				numberOfLabels);
		DataSet dataset = datasetIterator.next();
		return dataset;
	}
}
