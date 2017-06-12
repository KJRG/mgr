package io.github.kjrg.mgr;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.List;

import org.apache.poi.ss.usermodel.Cell;
import org.apache.poi.ss.usermodel.CellStyle;
import org.apache.poi.ss.usermodel.Font;
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

	private static final String REPORT_FILENAME_PREFIX = "results_";
	private static final String DATE_AND_TIME_FORMAT_FOR_REPORT_FILENAME = "yyyy_MM_dd_HH_mm_ss_SSS";
	private static final String XLSX_FILE_EXTENSION = ".xlsx";
	private static final int NUMBER_OF_COLUMNS_IN_RESULTS_SHEET = 6;
	private static final String EXPERIMENT_RESULTS_SHEET_NAME = "Results";

	/**
	 * Create report.
	 * 
	 * @param experimentResultList results of experiments
	 * @param outputDirectoryPath filepath of directory in which the report will be saved
	 * @throws FileNotFoundException
	 * @throws IOException
	 */
	public void createReport(List<ExperimentInfoDTO> experimentResultList, String outputDirectoryPath)
			throws FileNotFoundException, IOException {

		XSSFWorkbook workbook = new XSSFWorkbook();
		XSSFSheet sheet = workbook.createSheet(EXPERIMENT_RESULTS_SHEET_NAME);

		int numberOfRowsInHeader = createHeader(workbook, sheet);
		addExperimentResultsToReport(experimentResultList, sheet, numberOfRowsInHeader);
		autoSizeColumns(sheet, NUMBER_OF_COLUMNS_IN_RESULTS_SHEET);
		saveReport(workbook, createOutputFilepath(outputDirectoryPath));
		workbook.close();
	}

	private int createHeader(XSSFWorkbook workbook, XSSFSheet sheet) {
		int rowNumber = 0;

		Row row = sheet.createRow(rowNumber++);
		CellStyle headerStyle = createHeaderStyle(workbook);

		for (int i = 0; i < NUMBER_OF_COLUMNS_IN_RESULTS_SHEET; i++) {
			Cell cell = row.createCell(i);

			switch (i) {
			case 0:
				cell.setCellValue("Neurons in hidden layer");
				cell.setCellStyle(headerStyle);
				break;
			case 1:
				cell.setCellValue("Acivation function");
				cell.setCellStyle(headerStyle);
				break;
			case 2:
				cell.setCellValue("Updater");
				cell.setCellStyle(headerStyle);
				break;
			case 3:
				cell.setCellValue("F1 score");
				cell.setCellStyle(headerStyle);
				break;
			case 4:
				cell.setCellValue("Accuracy");
				cell.setCellStyle(headerStyle);
				break;
			case 5:
				cell.setCellValue("Recall");
				cell.setCellStyle(headerStyle);
				break;
			}
		}

		return rowNumber;
	}

	private CellStyle createHeaderStyle(XSSFWorkbook workbook) {
		CellStyle headerStyle = workbook.createCellStyle();
		Font defaultFont = workbook.createFont();
		defaultFont.setBold(true);
		headerStyle.setFont(defaultFont);
		return headerStyle;
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
	
	private void autoSizeColumns(XSSFSheet sheet, int numberOfColumns) {
		for (int i = 0; i < numberOfColumns; i++) {
			sheet.autoSizeColumn(i);
		}
	}

	private String createOutputFilepath(String outputDirectoryPath) {
		LocalDateTime currentDateAndTime = LocalDateTime.now();
		DateTimeFormatter formatter = DateTimeFormatter.ofPattern(DATE_AND_TIME_FORMAT_FOR_REPORT_FILENAME);
		return outputDirectoryPath + File.separator
				+ REPORT_FILENAME_PREFIX + currentDateAndTime.format(formatter) + XLSX_FILE_EXTENSION;
	}

	private void saveReport(XSSFWorkbook workbook, String outputFilepath) throws FileNotFoundException, IOException {
		try (FileOutputStream outputStream = new FileOutputStream(outputFilepath)) {
			workbook.write(outputStream);
		}
	}
}
