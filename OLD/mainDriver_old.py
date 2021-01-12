
import pandas as pd
import os
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np

class austinBikeShareVsCrime():
    # region Initialize Experiment
    def __init__(self,
                 periodFreq=None,
                 maxCrimeDistance=None
                 ):

        # Set default values for variables (as described in assignment)
        if periodFreq is None:
            periodFreq = 'M'

        if maxCrimeDistance is None:
            maxCrimeDistance = 500

        # Assign Class Values
        self.periodFreq = periodFreq
        self.maxCrimeDistance = maxCrimeDistance

    #endregion

    #region Read and Format Data
    def readAndFormatData(self):
        # Austin Bike Share Stations
        austin_bikeshare_stations_df = pd.read_csv('Data/austin_bikeshare_stations.csv')

        # Austin Bike Share Trips
        austin_bikeshare_trips_df = pd.read_csv('Data/austin_bikeshare_trips.csv')

        # Crime/Dist DataFrame (Jason)
        mergedCrime1 = pd.read_csv('Data/crime_merged1.csv', sep='\t')
        mergedCrime2 = pd.read_csv('Data/crime_merged2.csv', sep='\t')

        crimeWDistance = pd.concat([mergedCrime1, mergedCrime2])


        # Format dates to Datetime
        austin_bikeshare_trips_df['start_time'] = pd.to_datetime(austin_bikeshare_trips_df['start_time'], format='%Y-%m-%d %H:%M:%S')
        crimeWDistance['Occurred Date'] = pd.to_datetime(crimeWDistance['Occurred Date'], format='%m/%d/%Y')

        # Create Periods
        austin_bikeshare_trips_df['Start Time (Period)'] = austin_bikeshare_trips_df['start_time'].dt.to_period(self.periodFreq)
        crimeWDistance['Occurred Date (Period)'] = crimeWDistance['Occurred Date'].dt.to_period(self.periodFreq)


        self.austin_bikeshare_stations_df = austin_bikeshare_stations_df
        self.austin_bikeshare_trips_df = austin_bikeshare_trips_df
        self.crimeWDistance = crimeWDistance

    #endregion

    #region Count Crimes Around Stations per Period
    def countCrimesAroundStation(self, station, plotImage=False):
        # Create Storage DataFrame
        crimeAroundStationDF = pd.DataFrame(columns=['stationID', 'timePeriod', 'numCrimeWithinXDistance'])

        # Count only Crimes within maxCrimeDistance of station
        temp = self.crimeWDistance[self.crimeWDistance[f'distance to {station}'] <= self.maxCrimeDistance]
        crimeAroundStationGroupedDF = temp.groupby('Occurred Date (Period)').count()['Incident Number'].rename(f'{station}')

        # Create new temp dataframe
        crimeAroundStationDF['timePeriod'] = crimeAroundStationGroupedDF.index
        crimeAroundStationDF['stationID'] = crimeAroundStationGroupedDF.name
        crimeAroundStationDF['numCrimeWithinXDistance'] = crimeAroundStationGroupedDF.values

        if plotImage:
            crimeAroundStationDF = crimeAroundStationDF.drop(['stationID'], axis=1)

            crimeAroundStationDF = crimeAroundStationDF.set_index(['timePeriod'])

            # Plot
            fig, ax1 = plt.subplots()
            color = 'tab:blue'
            ax1.set_xlabel('Date')
            ax1.set_ylabel('Total Crime')
            ax1.plot(crimeAroundStationDF.index,
                     crimeAroundStationDF['numCrimeWithinXDistance'].values,
                     color=color,
                     label='Total Crime'
                     )
            ax1.tick_params(axis='y')
            ax1.set_title(f'Crimes Around station {station}')
            ax1.legend(loc=2)

            plt.show()

        return crimeAroundStationDF

    def countCrimesAroundAllStations(self):
        # Get all unique stations
        uniqueStations = self.austin_bikeshare_stations_df['station_id'].unique()

        crimePerStationDF = pd.DataFrame(columns=['stationID', 'timePeriod', 'numCrimeWithinXDistance'])

        # For each station calculate number of crimes within maxCrimeDistance of station
        for station in uniqueStations:
            crimeAroundStationDF = self.countCrimesAroundStation(station)

            # Append temp dataframe to total dataframe
            crimePerStationDF = pd.concat([crimePerStationDF, crimeAroundStationDF])

        self.crimePerStationDF = crimePerStationDF

    #endregion

    #region Count Trips per Period
    def countRidesForStation(self, station, plotImage=False):
        # Create Storage DataFrame
        numberOfRidesFromStation = pd.DataFrame(columns=['stationID', 'timePeriod', 'prevTimePeriod', 'totalRides'])

        # Count number of rides from station per period
        numberOfRidesFromStationGrouped = self.austin_bikeshare_trips_df[self.austin_bikeshare_trips_df['start_station_id']==station].groupby('Start Time (Period)').count()['start_station_id'].rename(f'{station}')

        numberOfRidesFromStation['timePeriod'] = numberOfRidesFromStationGrouped.index
        numberOfRidesFromStation['prevTimePeriod'] = numberOfRidesFromStation['timePeriod'] - 1
        numberOfRidesFromStation['stationID'] = numberOfRidesFromStationGrouped.name
        numberOfRidesFromStation['totalRides'] = numberOfRidesFromStationGrouped.values

        if plotImage:
            numberOfRidesFromStation = numberOfRidesFromStation.drop(['stationID', 'prevTimePeriod'], axis=1)

            numberOfRidesFromStation = numberOfRidesFromStation.set_index(['timePeriod'])

            # Plot
            fig, ax1 = plt.subplots()
            color = 'tab:blue'
            ax1.set_xlabel('Date')
            ax1.set_ylabel('Total Rides')
            ax1.plot(numberOfRidesFromStation.index,
                     numberOfRidesFromStation['totalRides'].values,
                     color=color,
                     label='Total Rides'
                     )
            ax1.tick_params(axis='y')
            ax1.set_title(f'Total Rides From station {station}')
            ax1.legend(loc=2)

            plt.show()

        return numberOfRidesFromStation

    def countRidesForAllStation(self):
        # Get all unique stations
        uniqueStations = self.austin_bikeshare_stations_df['station_id'].unique()

        ridesPerStationDF = pd.DataFrame(columns=['stationID', 'timePeriod', 'prevTimePeriod', 'totalRides'])

        # For each station calculate number of rides from station for every period
        for station in uniqueStations:
            numberOfRidesFromStation = self.countRidesForStation(station)

            if len(numberOfRidesFromStation)==0:
                continue
            else:
                # Append temp dataframe to total dataframe
                ridesPerStationDF = pd.concat([ridesPerStationDF, numberOfRidesFromStation])

        self.ridesPerStationDF = ridesPerStationDF

    #endregion

    #region Merge Crime (previous period) and Ride DataFrames
    def mergeCrimeAndRideDataframes(self):
        crimeVsNumberRidesPerStationDF = pd.merge(self.crimePerStationDF,
                                                  self.ridesPerStationDF,
                                                  how='inner',
                                                  left_on=['stationID', 'timePeriod'],
                                                  right_on=['stationID', 'prevTimePeriod'],
                                                  suffixes=('_x', '')
                                                  )
        crimeVsNumberRidesPerStationDF = crimeVsNumberRidesPerStationDF.drop(crimeVsNumberRidesPerStationDF.filter(regex='_x$').columns.tolist(), axis=1)
        crimeVsNumberRidesPerStationDF = crimeVsNumberRidesPerStationDF.drop(['prevTimePeriod'], axis=1)
        crimeVsNumberRidesPerStationDF = crimeVsNumberRidesPerStationDF.rename(columns={'numCrimeWithinXDistance': 'numCrimeWithinXDistancePrevTimePeriod'})

        self.crimeVsNumberRidesPerStationDF = crimeVsNumberRidesPerStationDF

    # endregion

    #region Crime Effect on Rides
    def crimeEffectOnRidesForStation(self, station, printResults=False, plotImage=False, saveResults=False):
        # Filter Dataframe
        crimeVsNumberRidesPerStationDFFiltered = self.crimeVsNumberRidesPerStationDF[self.crimeVsNumberRidesPerStationDF['stationID'] == f'{station}']

        # Set Index
        crimeVsNumberRidesPerStationDFFiltered = crimeVsNumberRidesPerStationDFFiltered.set_index(['timePeriod'])

        # Drop stationID Column
        crimeVsNumberRidesPerStationDFFiltered = crimeVsNumberRidesPerStationDFFiltered.drop(['stationID'], axis=1)

        # Rename Columns
        crimeVsNumberRidesPerStationDFFiltered = crimeVsNumberRidesPerStationDFFiltered.rename(columns={'numCrimeWithinXDistancePrevTimePeriod': 'Total Crime (Prev Period)',
                                                                                                        'totalRides': 'Total Rides'
                                                                                                        }
                                                                                               )
        if plotImage:
            crimeVsNumberRidesPerStationDFFiltered.plot(y='Total Crime (Prev Period)')
            ax = crimeVsNumberRidesPerStationDFFiltered['Total Rides'].plot(secondary_y=True)
            ax.set_ylabel('Total Rides')

            plt.show()


        # Run Regression
        y = np.asarray(crimeVsNumberRidesPerStationDFFiltered['Total Rides'].astype(int))
        X = np.asarray(crimeVsNumberRidesPerStationDFFiltered['Total Crime (Prev Period)'].astype(int))
        X = sm.add_constant(X)

        model = sm.OLS(y, X)
        results = model.fit()

        if printResults:
            print(results.summary())

    def crimeEffectOnRidesForAllStations(self, printResults=False, plotImage=False, saveResults = False):
        # Get all unique stations
        uniqueStations = self.austin_bikeshare_stations_df['station_id'].unique()

        if saveResults:
            # Create Empty DataFrame
            columnNames = ['Alpha Estimate',
                           'Alpha Standard Error',
                           'Alpha T-Stat',
                           'Beta Estimate',
                           'Beta Standard Error',
                           'Beta T-Stat'
                           ]
            exportDF = pd.DataFrame(index=uniqueStations, columns=columnNames)

        for station in uniqueStations:
            # Filter Dataframe
            crimeVsNumberRidesPerStationDFFiltered = self.crimeVsNumberRidesPerStationDF[self.crimeVsNumberRidesPerStationDF['stationID'] == f'{station}']

            # Set Index
            crimeVsNumberRidesPerStationDFFiltered['date'] = crimeVsNumberRidesPerStationDFFiltered['timePeriod'].apply(
                lambda x: str(x) + '-01')
            crimeVsNumberRidesPerStationDFFiltered['timePeriod'] = pd.to_datetime(
                crimeVsNumberRidesPerStationDFFiltered['date'])
            crimeVsNumberRidesPerStationDFFiltered = crimeVsNumberRidesPerStationDFFiltered.set_index(['timePeriod'])

            # Drop stationID Column
            crimeVsNumberRidesPerStationDFFiltered = crimeVsNumberRidesPerStationDFFiltered.drop(['stationID'], axis=1)

            # Rename Columns
            crimeVsNumberRidesPerStationDFFiltered = crimeVsNumberRidesPerStationDFFiltered.rename(columns={'numCrimeWithinXDistancePrevTimePeriod': 'Total Crime (Prev Period)',
                                                                                                            'totalRides': 'Total Rides'
                                                                                                            }
                                                                                                   )
            if (len(crimeVsNumberRidesPerStationDFFiltered['Total Rides'])<=5):
                continue
            else:
                print(station, len(crimeVsNumberRidesPerStationDFFiltered))
                if saveResults:
                    currentPath = os.getcwd()
                    newDirPath = f"{currentPath}/Results"
                    if not os.path.exists(newDirPath):
                        os.mkdir(newDirPath)

                # Plot
                fig, ax1 = plt.subplots()

                color = 'tab:red'
                ax1.set_xlabel('Date')
                ax1.set_ylabel('Total Crime (Previous Period)')
                ax1.plot(crimeVsNumberRidesPerStationDFFiltered['Total Crime (Prev Period)'].index,
                         crimeVsNumberRidesPerStationDFFiltered['Total Crime (Prev Period)'].values,
                         color=color,
                         label='Total Crime (Prev Period)'
                         )
                ax1.tick_params(axis='y')
                ax1.set_title(f'Rides per month vs Crimes (prev month) for station {station}')
                ax1.legend(loc=2)

                ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

                color = 'tab:blue'
                ax2.set_ylabel('Total Rides')  # we already handled the x-label with ax1
                ax2.plot(crimeVsNumberRidesPerStationDFFiltered['Total Rides'].index,
                         crimeVsNumberRidesPerStationDFFiltered['Total Rides'].values,
                         color=color,
                         label='Total Rides'
                         )
                ax2.tick_params(axis='y')
                ax2.legend(loc=1)

                fig.tight_layout()  # otherwise the right y-label is slightly clipped

                if saveResults:
                    print('Saving Figure')
                    plt.savefig(f"{newDirPath}/Figures/{station}_figure")

                if plotImage:
                    plt.show()

                # Run Regression
                y = np.asarray(crimeVsNumberRidesPerStationDFFiltered['Total Rides'].astype(int))
                X = np.asarray(crimeVsNumberRidesPerStationDFFiltered['Total Crime (Prev Period)'].astype(int))
                X = sm.add_constant(X)

                model = sm.OLS(y, X)
                results = model.fit()

                if printResults:
                    print('-------------------')
                    print(f"Station: {station}")
                    print(results.summary())
                    print('-------------------')
                    print('\n')

                if saveResults:
                    exportDF.loc[station] = [results.params[0],
                                             results.bse[0],
                                             results.tvalues[0],
                                             results.params[1],
                                             results.bse[1],
                                             results.tvalues[1],
                                             ]

        if saveResults:
            print('Saving Results')
            exportDF.to_csv(f"{newDirPath}/SummaryResults/results.csv")
    #endregion

run=True
if run:
    test = austinBikeShareVsCrime(periodFreq="M", maxCrimeDistance=500)
    test.readAndFormatData()
    test.countCrimesAroundAllStations()
    test.countRidesForAllStation()
    test.mergeCrimeAndRideDataframes()
    test.crimeEffectOnRidesForAllStations(printResults=True, plotImage=True, saveResults=False)














