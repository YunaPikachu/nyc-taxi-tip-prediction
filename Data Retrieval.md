# Data Retrieval  

## Bucket Setup in GCP  
- Created a new bucket directly in **Google Cloud Platform (GCP)**.  
- Organized the bucket with dedicated folders to store **raw**, **cleaned**, and **processed** datasets.  

## Data Source  
- **Provider:** NYC Taxi & Limousine Commission (TLC)  
- **URL:** [NYC TLC Trip Record Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)  

## Data Retrieval Method  
- The dataset (~10GB) was retrieved using Python’s `urllib.request` module.  
- A **Linux Virtual Machine (VM)** in GCP was used to:  
  - Run the retrieval scripts  
  - Handle large-scale data ingestion  
- Retrieved files were stored in the **GCP bucket** for subsequent cleaning and processing.  



# Data Retrieval Script for NYC Taxi Data
```
from urllib.request import urlretrieve

# Each parque file is named using the pattern: "yellow_tripdata_<year>-<month>.parquet"
# The Python script loops through years (e.g., 2023, 2024) and months (1–12) to generate these filenames automatically.
# Define years and months
years_list = ['2023', '2024']
months_list = [str(m).zfill(2) for m in range(1, 13)]  # '01' to '12'

# Loop through each year and month to download files
for year in years_list:
    for month in months_list:
        filename = f"yellow_tripdata_{year}-{month}.parquet"
        url = f"https://d37ci6vzurychx.cloudfront.net/trip-data/{filename}"
        
        print(f"Downloading file: {filename}")
        urlretrieve(url, filename)

print("Data retrieval completed successfully.")
