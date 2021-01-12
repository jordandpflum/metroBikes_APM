
import pandas as pd
from datetime import date, timedelta
import os
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np
from sklearn.preprocessing import MinMaxScaler

pd.set_option('display.max_columns', 500)
# pd.set_option('display.max_rows', 1500)
pd.set_option('display.width', 1000)

INDEX_START_DATE = date(2013, 12, 21)
INDEX_END_DATE = date(2017, 7, 31)



class austinBikeShareVsCrime:
    # region Initialize Experiment
    def __init__(self):
        self.austin_bikeshare_stations_df = None
        self.austin_bikeshare_trips_df = None
        self.crimeWDistance = None
        self.austin_weather_df = None
    #endregion

    #region Read and Format Data
    def readAndFormatData(self):
        # Austin Bike Share Stations
        austin_bikeshare_stations_df = pd.read_csv('Data/austin_bikeshare_stations.csv')

        # Austin Bike Share Trips
        austin_bikeshare_trips_df = pd.read_csv('Data/austin_bikeshare_trips.csv')

        # Austin Weather Data
        austin_weather_df = pd.read_csv('Data/austin_weather.csv')

        # Crime/Dist DataFrame (Jason)
        mergedCrime1 = pd.read_csv('Data/crime_merged1.csv', sep='\t')
        mergedCrime2 = pd.read_csv('Data/crime_merged2.csv', sep='\t')

        crimeWDistance = pd.concat([mergedCrime1, mergedCrime2])


        # Format dates to Datetime
        austin_bikeshare_trips_df['start_time'] = pd.to_datetime(austin_bikeshare_trips_df['start_time'], format='%Y-%m-%d %H:%M:%S')
        crimeWDistance['Occurred Date'] = pd.to_datetime(crimeWDistance['Occurred Date'], format='%m/%d/%Y')
        austin_weather_df['Date'] = pd.to_datetime(austin_weather_df['Date'], format='%m/%d/%Y')

        #weather data time conversion

        # Store Dataframes
        self.austin_bikeshare_stations_df = austin_bikeshare_stations_df
        self.austin_bikeshare_trips_df = austin_bikeshare_trips_df
        self.crimeWDistance = crimeWDistance
        self.austin_weather_df = austin_weather_df

    #endregion


    #Write Functions Here


def oneHotEncodingDayOfWeek(df, df_date_column):
    day_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

    def is_day(x, day_index):
        if x == day_index:
            return 1
        else:
            return 0

    df['temp'] = pd.to_datetime(df[df_date_column]).dt.dayofweek
    for index, day in enumerate(day_of_week):
        df[f'is{day}'] = df['temp'].apply(lambda x: is_day(x, index))

    df = df.drop(columns=['temp'])

    return df

def oneHotEncodingMonth(df, df_date_column):
    months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']

    def is_month(x, day_index):
        if x == day_index:
            return 1
        else:
            return 0

    df['temp'] = df[df_date_column].apply(lambda x: int(str(x).split('-')[1]) - 1)
    for index, month in enumerate(months):
        df[f'is{month}'] = df['temp'].apply(lambda x: is_month(x, index))

    df = df.drop(columns=['temp'])

    return df

def binaryEncodingEvent(df, df_date_column, event_list):

    df['isEventful'] = 0
    df[df[df_date_column].isin(event_list)] = 1

    return df

def weatherFeatures(df_weather=pd.read_csv('austin_weather.csv')):
    index = pd.date_range(INDEX_START_DATE, INDEX_END_DATE - timedelta(days=1), freq='d')
    # d = pd.DataFrame(index)
    result = pd.DataFrame(index)
    result = result.set_index(index)

    df_weather['date'] = pd.to_datetime(df_weather['Date'])
    df_weather = df_weather.set_index('date')


    result['pressure_1d'] = df_weather['SeaLevelPressureAvgInches'].shift(1).rolling(1).mean()
    result['pressure_3d'] = df_weather['SeaLevelPressureAvgInches'].shift(1).rolling(3).mean()
    result['pressure_1w'] = df_weather['SeaLevelPressureAvgInches'].shift(1).rolling(7).mean()
    result['pressure_1m'] = df_weather['SeaLevelPressureAvgInches'].shift(1).rolling(31).mean()

    result['temperature_1d'] = df_weather['TempAvgF'].shift(1).rolling(1).mean()
    result['temperature_3d'] = df_weather['TempAvgF'].shift(1).rolling(3).mean()
    result['temperature_1w'] = df_weather['TempAvgF'].shift(1).rolling(7).mean()
    result['temperature_1m'] = df_weather['TempAvgF'].shift(1).rolling(31).mean()

    df_weather['HumidityAvgPercent'] = pd.to_numeric(df_weather['HumidityAvgPercent'], errors='coerce')

    result['humidity_1d'] = df_weather['HumidityAvgPercent'].shift(1).rolling(1).mean()
    result['humidity_3d'] = df_weather['HumidityAvgPercent'].shift(1).rolling(3).mean()
    result['humidity_1w'] = df_weather['HumidityAvgPercent'].shift(1).rolling(7).mean()
    result['humidity_1m'] = df_weather['HumidityAvgPercent'].shift(1).rolling(31).mean()

    result['WindAvgMPH_3DayWindow'] = pd.to_numeric(df_weather['WindAvgMPH'], errors='coerce').shift(1).rolling(3).mean()
    result['WindAvgMPH_1WeekWindow'] = pd.to_numeric(df_weather['WindAvgMPH'], errors='coerce').shift(1).rolling(7).mean()
    result['WindAvgMPH_1MonthWindow'] = pd.to_numeric(df_weather['WindAvgMPH'], errors='coerce').shift(1).rolling(31).mean()
    result['VisibilityAvgMiles_1DayWindow'] = pd.to_numeric(df_weather['VisibilityAvgMiles'], errors='coerce').shift(1).rolling(1).mean()
    result['VisibilityAvgMiles_3DayWindow'] = pd.to_numeric(df_weather['VisibilityAvgMiles'], errors='coerce').shift(1).rolling(3).mean()
    result['VisibilityAvgMiles_1WeekWindow'] = pd.to_numeric(df_weather['VisibilityAvgMiles'], errors='coerce').shift(1).rolling(7).mean()
    result['VisibilityAvgMiles_1MonthWindow'] = pd.to_numeric(df_weather['VisibilityAvgMiles'], errors='coerce').shift(1).rolling(31).mean()
    result['DewPointAvgF_1DayWindow'] = pd.to_numeric(df_weather['DewPointAvgF'], errors='coerce').shift(1).rolling(1).mean()
    result['DewPointAvgF_3DayWindow'] = pd.to_numeric(df_weather['DewPointAvgF'], errors='coerce').shift(1).rolling(3).mean()
    result['DewPointAvgF_1WeekWindow'] = pd.to_numeric(df_weather['DewPointAvgF'], errors='coerce').shift(1).rolling(7).mean()
    result['DewPointAvgF_1MonthWindow'] = pd.to_numeric(df_weather['DewPointAvgF'], errors='coerce').shift(1).rolling(31).mean()
    result['TempAvgF_1DayWindow'] = pd.to_numeric(df_weather['TempAvgF'], errors='coerce').shift(1).rolling(1).mean()
    result['TempAvgF_3DayWindow'] = pd.to_numeric(df_weather['TempAvgF'], errors='coerce').shift(1).rolling(3).mean()
    result['TempAvgF_1WeekWindow'] = pd.to_numeric(df_weather['TempAvgF'], errors='coerce').shift(1).rolling(7).mean()
    result['TempAvgF_1MonthWindow'] = pd.to_numeric(df_weather['TempAvgF'], errors='coerce').shift(1).rolling(31).mean()
    result['PrecipitationSumInchesAvg_1DayWindow'] = pd.to_numeric(df_weather['PrecipitationSumInches'],
                                                                   errors='coerce').shift(1).rolling(1).mean()
    result['PrecipitationSumInchesAvg_3DayWindow'] = pd.to_numeric(df_weather['PrecipitationSumInches'],
                                                                   errors='coerce').shift(1).rolling(3).mean()
    result['PrecipitationSumInchesAvg_1WeekWindow'] = pd.to_numeric(df_weather['PrecipitationSumInches'],
                                                                    errors='coerce').shift(1).rolling(7).mean()
    result['PrecipitationSumInchesAvg_1MonthWindow'] = pd.to_numeric(df_weather['PrecipitationSumInches'],
                                                                     errors='coerce').shift(1).rolling(31).mean()


    def isEvent(x):
        if x != ' ':
            return 1
        else:
            return 0

    result['isWeatherEvent_1d'] = df_weather['Events'].shift(1).apply(lambda x: isEvent(x))

    return result

def oneHotEncodingDayOfWeekOfYear(df, df_date_column):
    week_of_year = range(52)

    def is_week(x, week_index):
        if x == week_index:
            return 1
        else:
            return 0

    dfCopy = df.copy()
    dfCopy['temp'] = pd.to_datetime(dfCopy[df_date_column]).dt.weekofyear
    for index, week in enumerate(week_of_year):
        dfCopy[f'isWeek_{week}'] = dfCopy['temp'].apply(lambda x: is_week(x, index))
    dfCopy = dfCopy.drop(columns=['temp'])
    dfCopy = dfCopy.set_index([df_date_column])

    return dfCopy

def countCrimesAffectingMetroBikesFeatures(df):
    index = pd.date_range(INDEX_START_DATE, INDEX_END_DATE - timedelta(days=1), freq='d')
    d = pd.DataFrame(index)
    df['newIndex'] = pd.to_datetime(df['Report Date'])
    merged = pd.merge(d, df, how='left', left_on=0, right_on='newIndex')
    stations = list(df.columns)[31:-1]
    merged['min_distance'] = merged[stations].min(axis=1)

    result = pd.DataFrame(index)
    result = result.set_index(index)
    distances = [50, 100, 250, 500]

    # def detrend(row, beta):
    #     print('a')

    for distance in distances:

        merged_filtered = merged[merged['min_distance'] < distance]


        merged_filtered = merged_filtered.groupby('newIndex').count()['min_distance'].reset_index().set_index('newIndex')
        result[f'crime_1d_{distance}m'] = merged_filtered['min_distance'].shift(1).rolling(1).mean()
        result[f'crime_3d_{distance}m'] = merged_filtered['min_distance'].shift(1).rolling(3).mean()
        result[f'crime_1w_{distance}m'] = merged_filtered['min_distance'].shift(1).rolling(7).mean()
        result[f'crime_1m_{distance}m'] = merged_filtered['min_distance'].shift(1).rolling(31).mean()

    result = result.drop(columns=[0])

    return result

def miscFeatures(df):

    df['date'] = pd.to_datetime(df['start_time']).dt.date
    index = pd.date_range(INDEX_START_DATE, INDEX_END_DATE - timedelta(days=1), freq='d')
    d = pd.DataFrame(pd.to_datetime(index).date)
    merged = pd.merge(d, df, how='left', left_on=0, right_on='date')

    result = pd.DataFrame(index)
    result = result.set_index(index)

    # x1 = 0.1824
    # coef = 401.3402

    detrended = pd.DataFrame(index)
    detrended = detrended.set_index(index)

    def detrend(row, beta):
        return row[0] - (beta * row['index'])

    def detrend1(row, beta):
        return row['duration_minutes'] - (beta * row['index'])

    data = merged.groupby('date').count()[0].reset_index().reset_index().set_index('date')
    detrended['ridesDetrended'] = data.apply(lambda x: detrend(x, 0.1824), axis=1)
    data = merged.groupby('date').sum()['duration_minutes'].reset_index().reset_index().set_index('date')
    detrended['durationDetrended'] = data.apply(lambda x: detrend1(x, 6.0721), axis=1)


    result['averageRides_1d'] = detrended['ridesDetrended'].shift(1)
    result['averageDuration_1d'] = detrended['durationDetrended'].shift(1)
    result['averageRides_3d'] = detrended['ridesDetrended'].shift(1).rolling(3).mean()
    result['averageDuration_3d'] = detrended['durationDetrended'].shift(1).rolling(3).mean()
    result['averageRides_7d'] = detrended['ridesDetrended'].shift(1).rolling(7).mean()
    result['averageDuration_7d'] = detrended['durationDetrended'].shift(1).rolling(7).mean()
    result['averageRides_1m'] = detrended['ridesDetrended'].shift(1).rolling(31).mean()
    result['averageDuration_1m'] = detrended['durationDetrended'].shift(1).rolling(31).mean()

    result['totalRides'] = detrended['ridesDetrended']

    result = result.drop(columns=[0])



    # result['averageRides_1d'] = merged.groupby('date').count()[0].shift(1)
    # result['totalRides'] = merged.groupby('date').count()[0]
    # result['averageDuration_1d'] = merged.groupby('date').sum()['duration_minutes'].shift(1)
    # result['averageRides_3d'] = merged.groupby('date').count()[0].shift(1).rolling(3).mean()
    # result['averageDuration_3d'] = merged.groupby('date').sum()['duration_minutes'].shift(1).rolling(3).mean()
    # result['averageRides_7d'] = merged.groupby('date').count()[0].shift(1).rolling(7).mean()
    # result['averageDuration_7d'] = merged.groupby('date').sum()['duration_minutes'].shift(1).rolling(7).mean()
    # result['averageRides_1m'] = merged.groupby('date').count()[0].shift(1).rolling(31).mean()
    # result['averageDuration_1m'] = merged.groupby('date').sum()['duration_minutes'].shift(1).rolling(31).mean()
    #
    # result = result.drop(columns=[0])

    return result

def sigAustinEventsFeatures():
    # Read in Dataframe
    sigDatesAustin = pd.read_csv('dates.csv')
    # Format as DateTime
    sigDatesAustin['Date'] = pd.to_datetime(sigDatesAustin['Date'])
    sigDatesAustin = sigDatesAustin.set_index('Date')
    # Create Result DataFrame
    index = pd.date_range(INDEX_START_DATE, INDEX_END_DATE - timedelta(days=1), freq='d')
    result = pd.DataFrame(index=index)
    result['UT_Football_Game'] = sigDatesAustin['UT_Football']
    result['ACL'] = sigDatesAustin['ACL']
    result['Holidays'] = sigDatesAustin['Holidays']
    result['SXSW'] = sigDatesAustin['SXSW']
    result['UT_Football_Game'] = result['UT_Football_Game'].fillna(0)
    result['ACL'] = result['ACL'].fillna(0)
    result['Holidays'] = result['Holidays'].fillna(0)
    result['SXSW'] = result['SXSW'].fillna(0)
    return result

def obtain_dataset():
    df_crime = pd.read_csv('crime_merged.csv')
    df_weather = pd.read_csv('austin_weather.csv')
    df_bike = pd.read_csv('austin_bikeshare_trips.csv')

    df_crime_modified = countCrimesAffectingMetroBikesFeatures(df_crime)
    df_weather_modified = weatherFeatures(df_weather)
    df_bike_modified = miscFeatures(df_bike)
    df_dates = sigAustinEventsFeatures()

    # data = pd.concat([df_weather_modified, df_bike_modified, df_dates], axis=1)
    data = pd.concat([df_crime_modified, df_weather_modified, df_bike_modified, df_dates], axis=1)
    data = oneHotEncodingDayOfWeek(data, 0)
    data = oneHotEncodingMonth(data, 0)
    data = oneHotEncodingDayOfWeekOfYear(data, 0)

    # data = data.drop(columns=[0])

    return data


if __name__ == "__main__":
    a = obtain_dataset()
    print(a)
    a.to_csv('output_new_detrended.csv')
    # a = weatherFeatures()
    # print(a)
    # a = sigAustinEventsFeatures()
    # print(a)


    # a = miscFeatures(pd.read_csv('austin_bikeshare_trips.csv'))
    # print(a)