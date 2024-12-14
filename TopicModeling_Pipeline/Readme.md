#Currently working on.... 

# Topic Modeling Pipeline

## Overview
This project is a comprehensive pipeline designed for topic modeling on text data. The pipeline processes raw data, cleans it, trains various models (e.g., LDA, NMF, BERTopic), and evaluates their performance. The process leverages ZenML to orchestrate different steps including data ingestion, cleaning, model training, and evaluation.

## Prerequisites
Before running the pipeline, make sure you have the following installed:

- **Python** (3.8 or higher)
- **ZenML**
- **Pandas**
-  **numpy** 
- Required Python packages listed in `requirements.txt`

You can install these dependencies using:

```bash
pip install -r requirements.txt
```

## Project Structure
### The project directory is structured as follows:

- **/zen**: ZenML configuration files.
- **/Data**: Directory containing raw and cleaned data files.
- **/source**: Contains the core logic of the pipeline, including scripts for:
  - `data_cleaning.py`: Data cleaning processes (punctuation removal, number removal, whitespace normalization, etc.).
  - `evaluate_model.py`: Evaluation of the trained models.
  - `model_dev.py`: Development and configuration of models (e.g., LDA, BERTopic).
- **/steps**: Contains individual step scripts for:
  - `ingest.py`: Ingesting raw data.
  - `cleaning.py`: Cleaning the ingested data.
  - `model_train.py`: Training different models (LDA, NMF, BERTopic).
  - `evaluation.py`: Evaluating the models.
- **requirements.txt**: Lists the Python packages required for this project.
- **run_pipeline.py**: Script to run the entire pipeline.
- **train_pipeline.py**: ZenML pipeline configuration file that defines the workflow of the pipeline.

## How to Run the Pipeline

1. **Clone the Repository:**
   To get started with the pipeline, you need to clone the repository from GitHub.

   ```bash
   git clone https://github.com/your-username/topic-modeling-pipeline.git
   cd topic-modeling-pipeline
   ```
2. **Set Up Your ZenML Environment:**
   Initialize ZenML with the default environment. This setup includes installing the required ZenML environment and configuring it.
   ```bash
   zenml init
   zenml downgrade
   ```
3. **Running in zenml:**
   ```bash
   zenml up
   ```
