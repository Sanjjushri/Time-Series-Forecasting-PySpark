from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.sql.types import StructType,StructField,StringType,LongType,DoubleType,FloatType

import statsmodels.tsa.api as sm
import numpy as np
import pandas as pd

from pyspark.sql import SparkSession

import warnings
warnings.filterwarnings('ignore')

spark = SparkSession.builder.appName('TSF-noise-pollution').getOrCreate()

# df = pd.read_csv("/Users/sanjju/projects/datasets/noise-pollution/station_month.csv")

data = spark.read.format('csv').options(header='true', inferSchema='true').load('dataset/station_month.csv').select('Year', 'Station', 'DayLimit', 'Day')

schema = StructType([StructField('Station', StringType(), True),
                    StructField('DayLimit', DoubleType(), True),
                    StructField('Day', DoubleType(), True)])

selected_com = data.groupBy(['Station','DayLimit']).count().filter("DayLimit > 0").select("Station","DayLimit")
data_selected_store_departments = data.join(selected_com, ['Station','DayLimit'],'inner')

# selected_com = data['Station']
# data_selected_store_departments = data['Station']

# df.set_index('Year',inplace = True)

@pandas_udf(schema, PandasUDFType.GROUPED_MAP)

def holt_winters_time_series_udf(data):
  
    data.set_index('Year',inplace = True)
    time_series_data = data['Day']
    
    ##the model
    model_monthly = sm.ExponentialSmoothing(np.asarray(time_series_data),trend='add').fit()

    ##forecast values
    forecast_values = pd.Series(model_monthly.forecast(1),name = 'fitted_values')
    
    return pd.DataFrame({'Station': [str(data.Station.iloc[0])], 'DayLimit': [int(data.DayLimit.iloc[1])], 'Day': [forecast_values[0]]})

forecasted_spark_df = data_selected_store_departments.groupby(['Station', 'DayLimit']).apply(holt_winters_time_series_udf)


## to see the forecasted results
forecasted_spark_df.show(10)

print((forecasted_spark_df.count(), len(forecasted_spark_df.columns)))
