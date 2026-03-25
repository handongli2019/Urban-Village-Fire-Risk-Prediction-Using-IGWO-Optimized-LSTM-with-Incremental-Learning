# Urban Village Fire Risk Prediction

This repository contains the MATLAB implementation of a **Grey Wolf Optimizer (GWO) + LSTM** model with an **incremental learning** strategy for urban village fire risk prediction.

The repository is prepared for **code sharing and method demonstration**. The original research dataset contains sensitive information and is therefore **not publicly released**. To support reproducibility, this repository includes:

- the full MATLAB code used for the workflow,
- a sample Excel data structure for testing the pipeline,
- the questionnaire file used for expert risk scoring.

## Repository structure

```text
Urban-Village-Fire-Risk-Prediction/
├── README.md
├── .gitignore
├── CITATION.cff
├── main_GWO_LSTM_IL.m
├── sample_data/
│   └── matrices_example.xlsx
├── questionnaire/
│   └── fire_risk_questionnaire.docx
└── results_example/
```

## Method overview

The workflow implemented in this repository includes the following steps:

1. Load time-series matrix samples from Excel.
2. Standardize the input data.
3. Split the dataset into training, validation, and testing subsets.
4. Use Grey Wolf Optimizer (GWO) to search for suitable LSTM hyperparameters.
5. Train the final LSTM model.
6. Simulate online prediction with incremental learning and replay-based updates.
7. Export prediction results.

## Data format

The code expects an Excel file with the following structure:

- **Sheet 1 to Sheet 100**: each sheet contains one sample matrix of size **3 × 30**.
- **Sheet `Y`**: target values for the 100 samples.

In the example provided here:

- **3** = number of input features,
- **30** = number of time steps,
- **100** = number of samples.

If you want to run the code on your own data, keep the same structure or modify the script accordingly.

## Expert questionnaire and scoring logic

The fire-risk indicator values in the study were obtained using an expert questionnaire.
Experts scored each indicator on a **0–100 scale**, where a **higher score indicates a higher fire-risk level**. The questionnaire file is included in the `questionnaire/` folder for transparency of the scoring standard.

## How to run

1. Open MATLAB.
2. Set the current folder to this repository.
3. Make sure Deep Learning Toolbox is installed.
4. Run:

```matlab
main_GWO_LSTM_IL
```

The script will:

- search for LSTM hyperparameters,
- train the prediction model,
- perform incremental learning,
- save prediction outputs into `results_example/`.

## Notes on reproducibility

The original dataset used in the manuscript is confidential and cannot be shared publicly.
This repository therefore provides a **sample data structure** instead of the full raw dataset.
Researchers can reproduce the full pipeline by replacing the sample file with data organized in the same format.

## Suggested citation text

If you use this repository in academic work, please cite the corresponding paper and acknowledge the code source.

## Contact

For questions about the implementation or data format, please contact the corresponding author of the paper.
