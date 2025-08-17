## Data Overview  

Each parquet file contains ~19 columns and around 2.9 million records per month.  
Some columns include missing values (e.g., passenger_count, RatecodeID).  

Data quality issues handled:  
- Missing values addressed during data cleaning.  
- Outliers (e.g., negative fares or extreme trip distances) treated as well.  


## Data Cleaning Visualizations

### Fare Amount Before & After Cleaning
![Fare Amount](images/BeforeAfterCleaningFareAmount.png)

### Tip Amount Before & After Cleaning
![Tip Amount](images/BeforeAfterCleaningTipAmount.png)

### Trip Distance Before & After Cleaning
![Trip Distance](images/BeforeAfterCleaningTripDistance.png)
