# Emotion Classifier Web Service

This repository contains code for a web service that performs emotion classification on text inputs. It utilizes the FastAPI framework for hosting the model as a web service and provides a user-friendly interface for making predictions.

## Prerequisites

Before running the code, ensure you have the following:

- Python 3.10 installed
- Dependencies listed in the `requirements.txt` file installed
- Access to a MySQL database for storing logged data (credentials provided through environment variables)
- Set up a Heroku account for deployment

## Getting Started

1. Clone the repository to your local machine.

2. Install the required dependencies by running the following command:

3. Update the necessary environment variables in the `.env` file:

- `HOST`: MySQL host address
- `DATABASE`: Name of the MySQL database
- `DBUSERNAME`: MySQL username
- `PASSWORD`: MySQL password
- `MY_MODEL`: Type of model to be used (`logistic` or `roberta`)

4. Train or download the pre-trained model checkpoints and place them in the `ckpt/` directory. Ensure the correct model checkpoints are used based on the chosen model type.

5. Set up the MySQL database and table structure for logging. Refer to the database configuration in the `model.py` file.
   Set up the MySQL database and table structure for logging. Run the following SQL query to create the `log_data` table:

   ```
   CREATE TABLE log_data ( timestamp DATETIME, user_input TEXT, model_prediction VARCHAR(10), score DECIMAL(18, 16) );
   ```

6. Start the web service by running the following command:

   ```
   uvicorn main:app --reload
   ```

7. Access the web service at [http://localhost:8000](http://localhost:8000) in your web browser.

## API Endpoint

The web service provides a single API endpoint for making predictions:

- Endpoint: `/predict`
- Method: POST
- Request body: JSON object containing the `text` field
- Response: JSON object with the predicted emotion label and score

## User Interface

A simple web-based user interface is available to interact with the web service. The interface allows users to enter a text input and receive the predicted emotion label and score.

Access the deployed app at [http://nlp-cw2.herokuapp.com/](http://nlp-cw2.herokuapp.com/)

## Additional Information

For more details about the project's architecture, model types, and logging mechanism, refer to the source code files: `model.py`, `main.py`, `Procfile`, and `.github/workflows/build-deploy.yaml`.

## Model Checkpoint

The model checkpoins used in this project can be downloaded from the following links:

- [RoBERTa Model Checkpoint](https://drive.google.com/file/d/1OvW0IlVbe31WOQxk-oE_Kxj-BN9Cg6cp/view?ts=64690774)

Please ensure that the correct checkpoints are used based on the chosen model type.
