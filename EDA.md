
Notes:

    This file contains code adapted from Big Data Technologies lecture materials 
    by Professors Holowczak, Baruch College. 

    The code has been modified and extended by Yu-na Choi for educational and 
    portfolio purposes (nyc-taxi-tip-prediction project).



# ==============================
# Import required modules
# ==============================
from google.cloud import storage

from io import StringIO, BytesIO

import pandas as pd

import gzip

# Pandas display options
pd.set_option('display.float_format', '{:.2f}'.format)

pd.set_option('display.width', 1000)


# ==============================
# Perform EDA function
# ==============================
def perform_EDA(df: pd.DataFrame, filename: str):

    """
    perform_EDA(df : pd.DataFrame, filename : str)
    Accepts a dataframe and a text filename as inputs.
    Runs some basic statistics on the data and outputs to console.

    :param df: The Pandas dataframe to explore
    :param filename: The name of the data file
    :returns:
    """
    print(f"{filename} Number of records:")
    print(df.count())

    number_of_duplicate_records = df.duplicated().sum()
    # old way
    # number_of_duplicate_records = len(df) - len(df.drop_duplicates())
    print(f"{filename} Number of duplicate records: {number_of_duplicate_records}")

    print(f"{filename} Info")
    print(df.info())

    print(f"{filename} Describe")
    print(df.describe())

    print(f"{filename} Columns with null values")
    print(df.columns[df.isnull().any()].tolist())

    rows_with_null_values = df.isnull().any(axis=1).sum()
    print(f"{filename} Number of Rows with null values: {rows_with_null_values}")

    integer_column_list = df.select_dtypes(include='int64').columns
    print(f"{filename} Integer data type columns: {integer_column_list}")

    float_column_list = df.select_dtypes(include='float64').columns
    print(f"{filename} Float data type columns: {float_column_list}")


# ==============================
# Main function for taxi data
# ==============================
def main_taxi():

    # This function is for the For-Hire Vehicle or Yellow Taxi data sets.
    # The store-and-forward-flag column is difficult to work with and should be dropped.

    categorical_columns_list = [
        "passenger_count", "PULocationID", "DOLocationID", "payment_type"
    ]

    # Change the bucket name to match your project bucket
    bucket_name = 'my-bigdata-project-yuna'

    # Create a client object that points to GCS
    storage_client = storage.Client()

    # Get a list of the 'blobs' (objects or files) in the bucket
    blobs = storage_client.list_blobs(bucket_name, prefix="landing/")

    # Iterate through the list and print out their names
    parquet_blobs = [blob for blob in blobs if blob.name.endswith('.parquet')]

    for blob in parquet_blobs:
        print(f"{blob.name} with size {blob.size} bytes created on {blob.time_created}")

        # Read in the Parquet file from the blob
        # Note the use of BytesIO and .download_as_bytes() function
        df = pd.read_parquet(BytesIO(blob.download_as_bytes()))

        perform_EDA(df, blob.name)

        # Gather the statistics on numeric columns
        numeric_summary_df = perform_EDA_numeric(df, blob.name)
        print(numeric_summary_df.head(24))

        # Gather statistics on the categorical columns
        categorical_summary_df = perform_EDA_categorical(
            df, blob.name, categorical_columns_list
        )
        print(categorical_summary_df.head(24))

        # For testing one file only
        if blob.name == "landing/yellow_tripdata_2022-01.parquet":
            break

if __name__ == "__main__":

    main_taxi()
