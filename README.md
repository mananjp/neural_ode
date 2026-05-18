# Sepsis Risk Prediction with Neural ODEs & LLM Explanations

This repository contains a full pipeline for predicting the onset of sepsis in ICU patients using continuous-time Neural Ordinary Differential Equations (Neural ODEs). It predicts whether a patient will develop sepsis within the next 6 hours based on a 24-hour observation window of vitals and lab results. 

Additionally, a Streamlit web application is provided that not only predicts the risk but also leverages a Large Language Model (Llama-3-70B via Groq) to generate clear, clinician-friendly explanations of the patient's risk profile.

## 📊 Model Architecture

The primary model used in this project is a **Neural ODE (v1)**, which natively handles continuous-time dynamics. 

### Neural ODE (v1) Structure:
*   **Input Dimension**: 41 (vitals and lab measurements).
*   **Encoder**: Maps the last observed time step into an initial latent state $z_0$. 
    *   `Linear(41, 64) -> ReLU -> Linear(64, 32)`
*   **ODE Solver**: Evolves the latent state continuously over time using a 4th-order Runge-Kutta (`rk4`) method. 
    *   `ODEFunc`: `Linear(32, 64) -> Softplus -> Linear(64, 64) -> Softplus -> Linear(64, 32)`
*   **Classifier**: Maps the final evolved latent state $z_T$ to a sepsis risk logit.
    *   `Linear(32, 64) -> ReLU -> Linear(64, 1)`

*Note: An LSTM baseline and an advanced GRU-ODE hybrid (v2) were also trained, but the standard Neural ODE (v1) was selected as it demonstrated the best overall generalization and AUROC.*

## 📈 Evaluation Metrics

The Neural ODE model was evaluated on a held-out validation set of patient time-series windows:
*   **AUROC (Area Under the Receiver Operating Characteristic curve)**: `0.7448`
*   **AUPRC (Area Under the Precision-Recall Curve)**: `0.0760`
*   **Brier Score**: `0.0402`

**Performance at Threshold = 0.10:**
*   **Sensitivity (Recall)**: `0.334`
*   **Specificity**: `0.911`
*   **Precision**: `0.097`
*   **F1-Score**: `0.150`

## 🛠️ Data Preprocessing & Generation

1.  **Imputation**: Missing values are handled per-patient using forward-filling followed by backward-filling. Any remaining `NaN` values are globally mean-imputed.
2.  **Normalization**: All 41 features are normalized using global means and standard deviations (zero mean, unit variance).
3.  **Windowing**: The data is chopped into 24-hour observation windows. The label is `1` if a sepsis event occurs anywhere within the following 6-hour horizon.

## 🚀 Usage

### 1. Requirements

Install the project dependencies:
```bash
pip install -r requirements.txt
```

To enable LLM explanations in the Streamlit app, create a `.env` file in the root directory and add your Groq API key:
```env
GROQ_API_KEY=your_groq_api_key_here
```

### 2. Command Line Inference (`sepsis_inference.py`)

You can run predictions directly from the command line on a dataset or a specific patient:

```bash
# Predict for a specific patient in a large dataset:
python sepsis_inference.py --model-path ./neural_ode_sepsis.pt --data-path ./Dataset.csv --patient-id 3

# Predict for a single-patient CSV:
python sepsis_inference.py --model-path ./neural_ode_sepsis.pt --data-path ./patient_3.csv
```

### 3. Streamlit Web App (`streamlit_sepsis_app.py`)

A fully interactive UI that allows you to upload CSVs, select patients, or manually input a single time snapshot. 

```bash
streamlit run streamlit_sepsis_app.py
```

**Features of the App:**
*   Load and parse patient datasets directly in the browser.
*   View formatted patient demographics and 24-hour vitals summary (Minimum, Maximum, Latest, and Trend).
*   Run the Neural ODE model to get a sepsis risk probability score.
*   Receive an automated, clinician-friendly paragraph explaining the prediction, generated securely via the Groq API (Llama 3).

## 📁 Repository Contents

*   `neural-ode (3).ipynb` - The original Jupyter Notebook containing the data pipeline, PyTorch model definitions, and training/evaluation loops.
*   `neural_ode_sepsis.pt` - The trained Neural ODE v1 model weights and preprocessing metadata.
*   `sepsis_inference.py` - CLI script for rapid headless predictions.
*   `streamlit_sepsis_app.py` - Interactive web interface.
*   `Dataset.csv` - The raw PhysioNet-style patient dataset (tracked via Git LFS).
*   `requirements.txt` - Python dependencies.
