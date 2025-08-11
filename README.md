# nyc-taxi-tip-prediction
Predicting NYC taxi tip amounts using Python, Linux VM, PySpark, GCP Dataproc, and Google Cloud Storage
Completed as part of CIS 9760: Big Data Technologies (Baruch College).

Tools & Technologies
1. **Linux VM**— retrieved and prepared ~10 GB of raw taxi trip data from Google Cloud Storage
2. **Google Cloud Storage (GCS)** — hosted dataset in Parquet format
3. **Google Cloud Dataproc** — ran distributed PySpark jobs for EDA, cleaning, and modeling
4. **PySpark** — feature engineering, regression model training, evaluation
5. **Python** — supplemental analysis
6. **Matplotlib / Seaborn** — data visualization and model evaluation plots

Process
1. **Data Retrieval**  
   - Used Linux VM to retrieve ~10 GB of TLC data and save it to the bucket on GCP.
   
2. **Data Cleaning**
   - Used (JupyterLab, GCP)
   - Removed rows with missing values and extreme outliers.
   - Filtered unrealistic trip distances and fares.
   
4. **Feature Engineering**
   -Used Pyspark
   - Extracted features from pickup datetime (hour, day of week, month).
   - Included trip distance, passenger count, and weather factors.
   
6. **Model Training & Evaluation**  
   - Used PySpark MLlib regression model.
   - Achieved RMSE of **2.35** on test data.

Results
- RMSE: **2.35**
- Key predictors of tip amount:
  - Trip distance
  - Time of day
  - Day of week

