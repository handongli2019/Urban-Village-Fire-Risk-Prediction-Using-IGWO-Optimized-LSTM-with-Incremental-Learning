
# Urban Village Fire Risk Prediction Using IGWO-Optimized LSTM with Incremental Learning

This repository provides the **MATLAB implementation** of the proposed **IGWO-LSTM model with Incremental Learning** for predicting fire risk in urban villages.

The framework integrates:

- **Improved Grey Wolf Optimizer (IGWO)** for hyperparameter optimization
- **Long Short-Term Memory (LSTM)** for time-series prediction
- **Incremental Learning (IL)** for adaptive model updating under changing fire risk conditions

The goal is to improve prediction accuracy while maintaining adaptability to evolving environmental and operational conditions in urban villages.

---

# Repository Structure

Urban-Village-Fire-Risk-Prediction-Using-IGWO-Optimized-LSTM-with-Incremental-Learning

│  
├── main_GWO_LSTM_IL.m              Main MATLAB script  
├── sample_data/                   Example dataset format  
│   └── matrices_example.xlsx  
│  
├── questionnaire/                 Expert scoring questionnaire  
│   └── fire_risk_questionnaire.docx  
│  
├── results_example/               Example output description  
│   └── README.md  
│  
├── CITATION.cff                   Citation information  
└── README.md

---

# Method Overview

The proposed framework consists of three key modules.

## 1. IGWO Hyperparameter Optimization

The **Improved Grey Wolf Optimizer (IGWO)** is used to optimize key LSTM hyperparameters:

- Number of hidden units
- Learning rate
- Training epochs

IGWO improves convergence efficiency and search capability compared to traditional optimization approaches.

---

## 2. LSTM Prediction Model

The **Long Short-Term Memory (LSTM)** network captures temporal dependencies in fire risk indicators.

Model architecture:

Input Layer → LSTM Layer → Dropout Layer → Fully Connected Layer → Regression Layer

---

## 3. Incremental Learning

To address dynamic changes in fire risk environments, an **incremental learning mechanism** is introduced.

The model:

- Monitors prediction error drift
- Updates model parameters when performance degradation is detected
- Uses replay samples to prevent catastrophic forgetting

This allows the model to remain effective as new data becomes available.

---

# Data Availability

The original dataset used in this study contains **sensitive investigation data related to urban village fire risk assessment** and therefore **cannot be publicly released**.

To ensure reproducibility:

- A **sample dataset structure** is provided in the `sample_data` folder.
- Researchers can reproduce the algorithm by replacing the example data with their own datasets following the same format.

---

# Data Format

Each sample is represented as a **3 × 30 matrix**:

- 3 → fire risk indicators
- 30 → time steps
- 100 → samples

Example structure used in the MATLAB code:

X = zeros(3,30,100)

Target values are stored in the Excel sheet named **Y**.

---

# Expert Questionnaire

Fire risk indicators were evaluated using an **expert scoring questionnaire**.

Experts were asked to assign a score between **0 and 100** for each indicator:

Higher score → Higher fire risk level

The questionnaire template is available in:

questionnaire/fire_risk_questionnaire.docx

---

# Requirements

The code was tested using:

MATLAB R2022a or later  
Deep Learning Toolbox

---

# How to Run

1. Open MATLAB  
2. Navigate to the repository directory  
3. Run the main script:

main_GWO_LSTM_IL

The script will:

- Load the dataset
- Train the IGWO-LSTM model
- Perform incremental learning updates
- Export prediction results

---

# Output Files

The program generates the following files:

GWO-LSTM.xlsx  
error_GWO-LSTM.xlsx  
GWO-LSTM-IL_Pred.xlsx  
GWO-LSTM-IL_Error.xlsx

These contain prediction results and model error analysis.

---

# Citation

If you use this code in your research, please cite our paper.

Citation information is provided in:

CITATION.cff

---

# License

This repository is released for **academic and research purposes only**.
