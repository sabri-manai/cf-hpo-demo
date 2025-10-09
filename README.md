# Hyperparameter Optimization & Counterfactual Explanations

## How to Run

### 1. Install Requirements

You may want to create a virtual environment:

```bash
python -m venv xai_hpo
source xai_hpo/bin/activate  # or xai_hpo\Scripts\activate on Windows
```

Use the provided `requirements.txt` to install all necessary dependencies:

```bash
pip install -r requirements.txt
```

### 2. Open the Notebook

Run the notebook:  
```bash
jupyter notebook hpo_adult.ipynb
```

You should always start by running the **Setup** section.
Then you can jump to Training Surrogate Models section.

---

## Notebook Structure

### Setup
Import all required packages. This section must always be run first.

### Data Collection
Use **Optuna** to tune **XGBoost** on the Adult dataset.  
We collect a dataset of hyperparameter configurations and resulting model metrics.
Data is persisted in local db file that will be created when you run the Optuna study for the first time.

### Exploring Collected Data
Visualize the distribution of collected parameters and performance scores.

### Cleaning Dataset
Remove duplicate categories and handle missing/null values.

### Feature Engineering (Soft Parameters)
Enhance the dataset by adding soft parameters like:
- Labeled ratio
- Missing rate
- Risk preference
- Decision speed

### Training Surrogate Models
Train and evaluate multiple surrogate models (e.g., Random Forest)  
to predict **F1-score** from both hard and soft parameters. 
You can run this directly after setup as clean dataset is provided in the data folder.

### Generating Counterfactuals
Use a [DiCE](https://github.com/interpretml/DiCE) explainer to generate counterfactuals.  
Specify goals like:
- "Achieve F1 ≥ 0.8"
- "Lower risk preference"
The model suggests what to change in the hyperparameter setup to reach these goals.

### Evaluating Counterfactuals
Compare generated counterfactuals with real XGBoost runs to assess their quality and reliability.

---


