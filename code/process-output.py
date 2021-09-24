import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import seaborn as sns
sns.set()

if __name__ == '__main__':
    def mean_confidence_interval(data, confidence=0.90):
        a = 1.0 * np.array(data)
        n = len(a)
        m, se = np.mean(a), scipy.stats.sem(a)
        h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
        return m, m-h, m+h

    n_experiments = 2
    n_replications = 2
    n_executions = n_experiments * n_replications

    statistics = (
        'totalUsers',
        'co2Avoided',
        'totalKm',
        'totalLostCustomers',
        'delayedUsers',
        'totalLostTime',
    )

    results = { f'{statistic}': [] for statistic in statistics }
    stations_population_data = []

    for exp in range(1, n_experiments + 1):
        for rep in range(1, n_replications + 1):
            file = open(f'../Modelo/MiBiciTuBici/model-output/output-exp{exp}-rep{rep}.json')
            data = json.load(file)
            stations_population_data.append(data['estaciones'])

            for statistic in statistics:
                results[statistic].append(data[statistic])

    print('-> calculating confidence intervals...')
    confidence_intervals = { f'{statistic}': None for statistic in statistics }
    for statistic, values in results.items():
        mean, lower, upper = mean_confidence_interval(values)
        confidence_intervals[statistic] = (lower, mean, upper)

    confidence_intervals = pd.DataFrame(confidence_intervals).T
    confidence_intervals.columns = ('Cota inferior', 'Media', 'Cota superior')
    confidence_intervals = confidence_intervals.round(decimals=2)
    confidence_intervals.to_csv('../Modelo/MiBiciTuBici/processed-model-output/intervals.csv')

    print('-> generating summary of results...')
    pd.DataFrame(results).describe().T.to_csv('../Modelo/MiBiciTuBici/processed-model-output/stats.csv')


    n_stations = len(stations_population_data[0])
    stations_results = [[] for _ in range(n_stations)]

    for i in range(n_executions):
        for j in range(n_stations):
            stations_results[j].append(stations_population_data[i][j])

    for station_id in range(5):
        print(f'-> generating histogram for station {station_id}')
        station_name = stations_results[station_id][0]['name']
        station_code = stations_results[station_id][0]['stationCode']
        occupied_percentage_along_runs = []

        for exec_id in range(n_executions):
             occupied_percentage_along_runs.append(
                 np.array([xy[1] for xy in stations_results[station_id][exec_id]['percentageOfOccupation']['plainDataTable'][:-1]])
             )

        occupied_percentage_along_runs = np.array(occupied_percentage_along_runs, dtype=object).flatten()
        fig, ax = plt.subplots()
        ax.hist(occupied_percentage_along_runs, bins=5)
        ax.set_title(f"Estación {station_name} | ID: {station_code}")
        ax.set_xlabel('Porcentaje de ocupación (%)')
        ax.set_ylabel('Count')
        fig.savefig(f'../Modelo/MiBiciTuBici/processed-model-output/histogram-code{station_code}.pdf', dpi=100)

    print('Done.')
