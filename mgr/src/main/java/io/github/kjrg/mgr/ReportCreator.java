package io.github.kjrg.mgr;

import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.List;

import org.apache.poi.ss.usermodel.Cell;
import org.apache.poi.ss.usermodel.Row;
import org.apache.poi.xssf.usermodel.XSSFSheet;
import org.apache.poi.xssf.usermodel.XSSFWorkbook;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.Updater;

import io.github.kjrg.mgr.dto.ExperimentInfoDTO;

/**
 * Class responsible for creating experiment reports.
 * 
 * @author Krzysztof Ga³ka
 */
public class ReportCreator {

	private static final int NUMBER_OF_COLUMNS_IN_RESULTS_SHEET = 6;
	private static final String EXPERIMENT_RESULTS_SHEET_NAME = "Results";

	/**
	 * Create report.
	 * 
	 * @param experimentResultList results of experiments
	 * @param outputFilepath filepath for report
	 * @throws FileNotFoundException
	 * @throws IOException
	 */
	public void createReport(List<ExperimentInfoDTO> experimentResultList, String outputFilepath)
			throws FileNotFoundException, IOException {

		XSSFWorkbook workbook = new XSSFWorkbook();
		XSSFSheet sheet = workbook.createSheet(EXPERIMENT_RESULTS_SHEET_NAME);

		int rowNumber = createHeader(sheet);
		addExperimentResultsToReport(experimentResultList, sheet, rowNumber);
		saveReport(workbook, outputFilepath);
		workbook.close();
	}

	private int createHeader(XSSFSheet sheet) {
		int rowNumber = 0;

		Row row = sheet.createRow(rowNumber++);
		for (int i = 0; i < NUMBER_OF_COLUMNS_IN_RESULTS_SHEET; i++) {
			Cell cell = row.createCell(i);

			switch (i) {
			case 0:
				cell.setCellValue("Neurons in hidden layer");
				break;
			case 1:
				cell.setCellValue("Acivation function");
				break;
			case 2:
				cell.setCellValue("Updater");
				break;
			case 3:
				cell.setCellValue("F1 score");
				break;
			case 4:
				cell.setCellValue("Accuracy");
				break;
			case 5:
				cell.setCellValue("Recall");
				break;
			}
		}

		return rowNumber;
	}

	private void addExperimentResultsToReport(List<ExperimentInfoDTO> experimentResultList, XSSFSheet sheet,
			int headerRows) {
		int rowNumber = headerRows;

		for (ExperimentInfoDTO experimentResult : experimentResultList) {
			Row row = sheet.createRow(rowNumber++);
			addSingleExperimentResultToReport(experimentResult, row);
		}
	}

	private void addSingleExperimentResultToReport(ExperimentInfoDTO experimentResult, Row row) {
		for (int i = 0; i < NUMBER_OF_COLUMNS_IN_RESULTS_SHEET; i++) {
			Evaluation experimentEvaluation = experimentResult.getExperimentResult();
			if (experimentEvaluation == null) {
				return;
			}

			Cell cell = row.createCell(i);

			switch (i) {
			case 0:
				cell.setCellValue(experimentResult.getNumberOfNeuronsInHiddenLayer());
				break;
			case 1:
				cell.setCellValue(experimentResult.getActivationFunction());
				break;
			case 2:
				Updater updater = experimentResult.getUpdater();
				String updaterInfo = (updater == null ? "" : updater.toString());
				cell.setCellValue(updaterInfo);
				break;
			case 3:
				cell.setCellValue(experimentEvaluation.f1());
				break;
			case 4:
				cell.setCellValue(experimentEvaluation.accuracy());
				break;
			case 5:
				cell.setCellValue(experimentEvaluation.recall());
				break;
			}
		}
	}

	private void saveReport(XSSFWorkbook workbook, String outputFilepath) throws FileNotFoundException, IOException {
		try (FileOutputStream outputStream = new FileOutputStream(outputFilepath)) {
			workbook.write(outputStream);
		}
	}
}
