# Stock Prediction Project - Group 13

This project aims to predict stock prices using various machine learning models and technical indicators. It includes a complete pipeline for data extraction, transformation, model training, scoring, and backtesting.

## Project Structure

- `config/`: Contains configuration files for database and model parameters.
- `Scripts/`: Main execution scripts for different stages of the pipeline:
  - `extract.py`: Fetches data from external sources (e.g., Yahoo Finance).
  - `transformation.py`: Cleans and prepares data for training.
  - `train_models.py`: Trains machine learning models.
  - `scoring.py`: Evaluates model performance.
  - `backtesting.py`: Tests trading strategies based on predictions.
  - `main.py`: Orchestrates the entire pipeline.
- `Stock_Prediction_EDA.ipynb`: Exploratory Data Analysis notebook.
- `create_tables.sql`: SQL script to initialize the database schema.

## Setup Instructions

### 1. Prerequisites
- Python 3.8+
- PostgreSQL

### 2. Installation
Clone the repository and install dependencies:
```bash
git clone <repository-url>
cd Term_Project_Group13
pip install -r requirements.txt
```
*(Note: Ensure you have `python-dotenv` and other necessary libraries installed.)*

### 3. Environment Configuration
Create a `.env` file in the root directory based on `.env.example`:
```bash
DB_HOST=localhost
DB_PORT=5432
DB_NAME=stock_prediction_db
DB_USER=postgres
DB_PASSWORD=your_password_here
```

### 4. Database Setup
Run the `create_tables.sql` script in your PostgreSQL database to create the required tables.

### 5. Running the Project
You can run the main pipeline using:
```bash
python Scripts/main.py
```

## Contributing
- Partipciants: (See `Partipciants.txt` for details)
