# ML Prediction on Hydrogen Generation using Water Electrolysis

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)

This repository contains a Machine Learning project aimed at predicting the efficiency and rate of hydrogen production through water electrolysis. By leveraging operational data such as temperature, pressure, and voltage, the model provides real-time estimates for clean energy production metrics.

---

## Project Overview

Water electrolysis is a key technology for producing "Green Hydrogen." This project uses a **Multi-Output Random Forest Regression** model to predict two critical targets simultaneously based on real-world industrial sensor data.

### Targets
1. **H2 Flow Rate ($Nm^3/h$):** The volume of hydrogen gas produced per hour.
2. **Efficiency (%):** The overall operational efficiency of the electrolysis stack.

---

## Dataset Description

The analysis is based on the `Hydrogen Generation.csv` dataset, which captures several operational variables:

| Feature | Description |
| :--- | :--- |
| `current_A` | Electrical current supplied (Amperes) |
| `voltage_V` | Operational voltage (Volts) |
| `stack_temp_C` | Temperature of the electrolysis stack (°C) |
| `anode_p_kPa` / `cathode_p_kPa` | Pressure levels at the electrodes (kPa) |
| `water_flow_Lmin` | Rate of water supply (L/min) |
| `ambient_temp_C` / `ambient_rh_pct` | External environmental conditions |

---

## Technical Implementation

### Preprocessing
* **Missing Value Imputation:** Filled using median values to maintain data distribution stability.
* **Feature Selection:** Dropped temporal metadata to focus purely on physical and electrical predictors.
* **Train-Test Split:** Data divided into 80% training and 20% testing sets.

### Architecture
* **Algorithm:** Random Forest Regressor.
* **Wrapper:** `MultiOutputRegressor` to handle multiple target variables independently but within a unified framework.
* **Storage:** Model exported using `joblib` for easy deployment.

---

## Performance & Results

The model achieved high accuracy on the test set:

| Metric | H2 Flow Rate ($Nm^3/h$) | Efficiency (%) |
| :--- | :--- | :--- |
| **MAE** | 0.009 | 1.802 |
| **$R^2$ Score** | **0.827** | **0.911** |

### Insights
* **Efficiency** prediction is highly robust with an $R^2$ of **0.91**.
* Operational **Current** and **Stack Temperature** were identified as the most significant features influencing production.

---

## How to Use

### 1. Requirements
Install the necessary libraries:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib
```

### 2. Run the Analysis
Open and run `Main file.ipynb` in a Jupyter environment to train the model and see visualization plots (Heatmaps, Distributions).

### 3. Inference
You can load the saved model `hydrogen_multioutput_model.joblib` and use the built-in prediction function:
```python
import joblib
# Load the model bundle
bundle = joblib.load("hydrogen_multioutput_model.joblib")
model = bundle["model"]
# Predict based on feature input
prediction = model.predict([features_list])
```

---

## Author
**Jatin Gupta**  
[GitHub Profile](https://github.com/Official-JK11)

---
*Developed as part of an investigation into Sustainable Energy Solutions.*