import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import config
 # Assuming config.py is in the same directory as this script
import logging

# logging configure

logger = logging.getLogger('data_ingestion')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('errors.log')
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_data(data_url: str) -> pd.DataFrame:
    """
    Load data from a CSV file.

    Args:
        data_url (str): The URL or path to the CSV file.

    Returns:
        pd.DataFrame: The loaded DataFrame.
    """
    try:
        df = pd.read_csv(data_url)
        logger.info("Data loaded successfully from %s", data_url)
        return df
    except pd.errors.ParserError as e:
        logger.error("Error parsing CSV file at %s: %s", data_url, e)
        raise
    except Exception as e:
        logger.error("Error loading data from %s: %s", data_url, e)
        raise


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the DataFrame by dropping unnecessary columns and filtering sentiment labels.
    Args:
        df (pd.DataFrame): The DataFrame to preprocess.
    Returns:
        pd.DataFrame: The preprocessed DataFrame.
    """

    try:
        df.drop(columns=['tweet_id'], inplace=True)
        label_categories = ['happiness', 'sadness']
        final_df = df[df['sentiment'].isin(label_categories)]
        final_df['sentiment'].replace({'happiness': 1, 'sadness': 0}, inplace=True)
        logger.info("Data preprocessed successfully")
        return final_df

    except KeyError as e:
        logger.error("KeyError during preprocessing: %s", e)
        raise
    except Exception as e:
        logger.error("Error preprocessing data: %s", e)
        raise

    
def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, output_dir: str):
    """
    Save the training and testing data to CSV files.
    
    Args:
        train_data (pd.DataFrame): The training data.
        test_data (pd.DataFrame): The testing data.
        output_dir (str): The directory to save the files.
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        train_data.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
        test_data.to_csv(os.path.join(output_dir, 'test.csv'), index=False)
        logger.info("Data saved successfully to %s", output_dir)
    except Exception as e:
        logger.error("Error saving data: %s", e)
        raise


def main():
    """
    Main function to execute the data ingestion and preprocessing pipeline.
    """
    try:
        # Load configuration
        data_url = config.DATA_URL
        output_dir = config.OUTPUT_DIR

        # Load data
        df = load_data(data_url)
        # Preprocess data
        preprocessed_df = preprocess_data(df)
        # Split data into training and testing sets
        train_df, test_df = train_test_split(preprocessed_df, test_size=0.2, random_state=42, stratify=preprocessed_df['sentiment'])
        # Save the processed data
        save_data(train_df, test_df, output_dir)

    except Exception as e:
        logger.error("An error occurred in the main function: %s", e)


if __name__ == "__main__":
    main()