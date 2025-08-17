# Data Overview
-Each parquet file contains ~19 columns.
-On average, each monthly file (e.g., yellow_tripdata_2023-01.parquet) has ~2.9 million records.
-Some columns include missing values:
  passenger_count, RatecodeID, congestion_surcharge, and Airport_fee each have around 140K null records in January 2023.

# Data quality issues handled:
-Missing values are addressed during the Data Cleaning stage.
-Outliers (e.g., extreme values like negative fares or abnormally large trip distances) are identified and treated during the Data Cleaning stage as well.


