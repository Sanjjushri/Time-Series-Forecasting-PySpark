from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.sql.types import StructType,StructField,StringType,LongType,DoubleType,FloatType

import statsmodels.tsa.api as sm
import numpy as np
import pandas as pd

from pyspark.sql import SparkSession

spark = SparkSession.builder.appName('TSF-pollution').getOrCreate()


data = spark.read.format('csv').options(header='true', inferSchema='true').load('dataset/air_quality_index.csv').select('DATE', 'COUNTRY','CITY','VALUE')

schema = StructType([StructField('COUNTRY', StringType(), True),
                     StructField('CITY', StringType(), True),
                     StructField('VALUE', DoubleType(), True)])

selected_com = data.groupBy(['COUNTRY','CITY']).count().filter("count > 104").select("COUNTRY","CITY")
data_selected_store_departments = data.join(selected_com, ['COUNTRY','CITY'],'inner')

# df.set_index('DATE',inplace = True)

@pandas_udf(schema, PandasUDFType.GROUPED_MAP)


def holt_winters_time_series_udf(data):
  
    data.set_index('DATE',inplace = True)
    time_series_data = data['VALUE']
    
    ##the model
    model_monthly = sm.ExponentialSmoothing(np.asarray(time_series_data),trend='add').fit()

    ##forecast values
    forecast_values = pd.Series(model_monthly.forecast(1),name = 'fitted_values')
    
    return pd.DataFrame({'COUNTRY': [str(data.COUNTRY.iloc[0])],'CITY': [str(data.CITY.iloc[1])],'VALUE': [forecast_values[0]]})

forecasted_spark_df = data.groupby(['COUNTRY','CITY']).apply(holt_winters_time_series_udf)

forecasted_spark_df.show(10)

print((forecasted_spark_df.count(), len(forecasted_spark_df.columns)))

