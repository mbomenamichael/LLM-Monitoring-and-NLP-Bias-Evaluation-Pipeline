# LLM Monitoring and NLP Bias Evaluation Pipeline

## Overview

This project implements a pipeline for monitoring and analysing bias in Large Language Model (LLM) hiring evaluations.

The system processes candidate CVs, submits them to an LLM for evaluation, and analyses the results for potential bias across demographic groups. Statistical fairness metrics and significance testing are used to measure disparities in hiring decisions and competence scores.

The project demonstrates how Natural Language Processing, LLM inference, and statistical fairness analysis can be combined to audit AI decision-making systems.

## Tech Stack

### Languages & Libraries

- Python
- Hugging Face Transformers (RoBERTa)
- OpenAI API
- Pandas
- NumPy
- SciPy
- Matplotlib
- MLflow

## Concepts Used

- Natural Language Processing (NLP)
- Large Language Model evaluation
- Sentiment analysis
- Fairness and bias detection in AI systems
- Statistical hypothesis testing
- Bootstrap resampling
- Experiment tracking (MLflow)

### Data & Infrastructure

- AWS S3 (dataset storage)
- Excel datasets
- API-based model inference

## System Pipeline

Candidate CV Dataset
     ↓
Data Preprocessing & Text Extraction
     ↓
LLM Resume Evaluation (GPT-4 API)
     ↓
Hiring Decision & Competence Scoring
     ↓
Demographic Metadata Inference
     ↓
Fairness Metrics Calculation
     ↓
Statistical Testing & Bootstrap Analysis
     ↓
Experiment Tracking (MLflow)
     ↓
Bias Visualisation & Reporting

## How It Works

1. CV datasets are loaded and preprocessed to extract resume text and metadata.

2. Each candidate profile is evaluated using a Large Language Model through API inference, generating:
   - qualitative evaluation
   - binary hire decision
   - competence score.

3. Demographic attributes (such as gender and ethnicity) are inferred from candidate names.

4. Fairness metrics are computed to measure potential bias in model decisions.

5. Statistical testing and resampling techniques are applied to assess the significance of observed disparities.

6. Results are logged and visualised to analyse patterns in model behaviour across demographic groups.

## Running the Project

Install dependencies:

pip install pandas numpy scipy matplotlib transformers mlflow openai

Run the evaluation pipeline:

python testing_script.py

Run the comparison and batch evaluation pipeline:

python comparison_script.py

Generate fairness visualisations and bias analysis:

python visualisation_script.py

## Results

The pipeline produces several outputs:

- Candidate evaluation results (LLM reasoning, hiring decision, competence score)
- Group comparison rankings between candidates
- Fairness metrics across demographic groups
- Statistical significance tests for bias detection
- Visualisations showing score distributions and hiring rates

## Why This Project Matters

As LLMs are increasingly used in hiring and decision-making systems, understanding potential bias is critical.

This project demonstrates how AI systems can be audited using:

- NLP pipelines for processing candidate data
- LLM inference for automated evaluation
- statistical fairness metrics for bias detection
- experiment tracking and reproducible analysis

It highlights the importance of building transparent monitoring tools for responsible AI deployment.
