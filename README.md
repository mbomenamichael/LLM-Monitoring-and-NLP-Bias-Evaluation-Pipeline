![Python](https://img.shields.io/badge/Python-3.10-blue)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow)
![MLflow](https://img.shields.io/badge/MLflow-ExperimentTracking-blue)
![AWS](https://img.shields.io/badge/AWS-S3-orange)
![License](https://img.shields.io/badge/License-MIT-green)

# LLM Monitoring and NLP Bias Evaluation Pipeline

## Overview

This project implements an end-to-end pipeline for auditing and monitoring bias in Large Language Model (LLM) decision systems.

The system evaluates candidate CVs using an LLM, then analyses model outputs for demographic bias using statistical fairness metrics, hypothesis testing, and visualisation.

The project demonstrates how Natural Language Processing, LLM inference, and fairness evaluation techniques can be integrated into an AI monitoring workflow for responsible AI deployment.

It is designed as a lightweight experiment pipeline for analysing model behaviour, logging results, and identifying potential bias in automated decision-making systems.

## Tech Stack

### Languages & Libraries

- Python
- Pandas
- NumPy
- SciPy
- Matplotlib
- Hugging Face Transformers (RoBERTa)
- OpenAI / LLM API
- MLflow

### Data & Infrastructure

- AWS S3 (dataset storage)
- Excel / CSV datasets
- API-based model inference

## Concepts Used

- Natural Language Processing (NLP)
- Large Language Model evaluation
- Sentiment analysis
- AI fairness and bias detection
- Statistical hypothesis testing
- Bootstrap resampling
- Experiment tracking and model monitoring
- Responsible AI and algorithmic auditing

## System Pipeline

CV Dataset (Excel / CSV)
     ↓
Data Preprocessing (Pandas)
     ↓
LLM Resume Evaluation (API Inference)
     ↓
Hiring Decision + Competence Score
     ↓
Sentiment Analysis (RoBERTa)
     ↓
Fairness Metric Calculation
     ↓
Statistical Testing & Bootstrap Resampling
     ↓
Experiment Tracking (MLflow)
     ↓
Bias Visualisation & Monitoring

## How It Works

1. Candidate CV datasets are loaded and preprocessed using Python and Pandas.

2. Each CV is submitted to a Large Language Model through an API to generate:
   - hiring decision (binary outcome)
   - competence score
   - reasoning output.

3. Sentiment analysis is performed using a Hugging Face RoBERTa model to analyse tone and reasoning patterns in model outputs.

4. Demographic attributes (such as gender and ethnicity) are associated with candidate profiles to enable bias analysis.

5. Fairness metrics are calculated to identify disparities in hiring decisions across demographic groups.

6. Statistical testing and bootstrap resampling are applied to determine whether observed differences are statistically significant.

7. Experiments are logged using MLflow, and visualisations are generated to analyse model behaviour and bias patterns.

## Running the Project

Install dependencies:

pip install pandas numpy scipy matplotlib transformers mlflow openai

Run the LLM evaluation pipeline:

python testing_script.py

Run the candidate comparison experiments:

python comparison_script.py

Generate fairness metrics and bias visualisations:

python visualisation_script.py

## Results

The pipeline produces several outputs:

- Candidate evaluation results (LLM reasoning, hiring decision, competence score)
- Group comparison rankings between candidates
- Fairness metrics across demographic groups
- Statistical significance tests for bias detection
- Visualisations showing score distributions and hiring rates

## Why This Project Matters

As Large Language Models are increasingly used in automated decision systems, monitoring and auditing their behaviour is essential.

This project demonstrates how to build an AI monitoring pipeline that combines:

- NLP-based data processing
- LLM inference and evaluation
- fairness metrics and statistical testing
- experiment tracking and visualisation

## Author

**Michael Mbomena**

BSc Computer Science  
MSc Artificial Intelligence  

GitHub: https://github.com/mbomenamichael

The system illustrates how machine learning models can be analysed and monitored for bias, helping support responsible and transparent AI deployment.
