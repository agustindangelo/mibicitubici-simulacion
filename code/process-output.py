import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import seaborn as sns
import os
sns.set()

if __name__ == '__main__':
    model_path = '../Modelo/MiBiciTuBici'
    model_output_path = f'{model_path}/model-output'
    processed_output_path = f'{model_path}/processed-model-output/'
    experiment_1_path = f'{model_output_path}/experiment-1'
    experiment_2_path = f'{model_output_path}/experiment-2'

    # --> Sanity checks
    if not os.path.exists(model_output_path) or len(os.listdir(model_output_path)) == 0:
        raise Exception('No model output is available. Run the AnyLogic experiment and try again.')

    if not os.path.exists(experiment_1_path) or len(os.listdir(experiment_1_path)) == 0:
        raise Exception('No model output is available for experiment 1. Run the AnyLogic experiment and try again.')
    if not os.path.exists(experiment_2_path) or len(os.listdir(experiment_2_path)) == 0:
        raise Exception('No model output is available for experiment 2. Run the AnyLogic experiment and try again.')

    if not os.path.exists(processed_output_path):
        os.mkdir(processed_output_path)

    experiment_1_files = os.listdir(experiment_1_path)
    experiment_2_files = os.listdir(experiment_2_path)

    if len(experiment_1_files) != len(experiment_2_files):
        raise Exception('The amount of files for experiment 1 do not match with the amount for experiment 2. Aborting.')

    for file in experiment_1_files:
        if not file.startswith('output') and not file.endswith('.json'):
            raise Exception(f'File {file} is not recognized. Aborting.')

    for file in experiment_2_files:
        if not file.startswith('output') and not file.endswith('.json'):
            raise Exception(f'File {file} is not recognized. Aborting.')

    print(f'-> {len(experiment_1_files)} output files will be processed for each experiment.')
    # -----------------------

    def mean_confidence_interval(data, confidence=0.90):
        a = 1.0 * np.array(data)
        n = len(a)
        m, se = np.mean(a), scipy.stats.sem(a)
        h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
        return m, m-h, m+h

    n_replications = len(experiment_1_files)

    statistics = (
        'totalUsers',
        'co2Avoided',
        'totalKm',
        'totalLostCustomers',
        'delayedUsers',
        'totalLostTime',
    )

    results_exp_1 = { f'{statistic}': [] for statistic in statistics }
    results_exp_2 = { f'{statistic}': [] for statistic in statistics }
    results_difference = { f'{statistic}': [] for statistic in statistics }

    stations_population_data_exp_1 = []
    stations_population_data_exp_2 = []

    # Load files content in memory
    for output_file in experiment_1_files:
        with open(f'{model_output_path}/{output_file}') as f:
            data = json.load(f)

        stations_population_data_exp_1.append(data['estaciones'])

        for statistic in statistics:
            results_exp_1[statistic].append(data[statistic])

    for output_file in experiment_2_files:
        with open(f'{model_output_path}/{output_file}') as f:
            data = json.load(f)

        stations_population_data_exp_2.append(data['estaciones'])

        for statistic in statistics:
            results_exp_2[statistic].append(data[statistic])

    # -----------------------------

    # --> Confidence intervals for the difference
    print('-> calculating confidence intervals for the difference of the realizations...')
    confidence_intervals = { f'{statistic}': None for statistic in statistics }

    for statistic in statistics:
        results_difference[statistic].append(np.array(results_exp_1[statistic]) - np.array(results_exp_2[statistic]))

    for statistic, values in results_difference.items():
        mean, lower, upper = mean_confidence_interval(values)
        confidence_intervals[statistic] = (lower, mean, upper)

    confidence_intervals = pd.DataFrame(confidence_intervals).transpose
    confidence_intervals.columns = ('Cota inferior', 'Media', 'Cota superior')
    confidence_intervals = confidence_intervals.round(decimals=2)
    confidence_intervals.to_csv('../Modelo/MiBiciTuBici/processed-model-output/intervals.csv')
    # ------------------------------------------

    # --> Summary of results for all statistics of interest
    print('-> generating summary of results...')
    summary_exp_1 = pd.DataFram(results_exp_1).describe().transpose.round(decimals=2)
    summary_exp_2 = pd.DataFram(results_exp_2).describe().transpose.round(decimals=2)
    summary_exp_1.to_csv('../Modelo/MiBiciTuBici/processed-model-output/exp-1-summary.csv')
    summary_exp_2.to_csv('../Modelo/MiBiciTuBici/processed-model-output/exp-2-summary.csv')
    # -----------------------------------------------------

    # Per-station analysis
    n_stations = len(stations_population_data_exp_1[0])
    stations_results_exp_1 = [[] for _ in range(n_stations)]
    stations_results_exp_2 = [[] for _ in range(n_stations)]
    occupation_values_per_station_along_runs = [[],[]] # for [experiment_1, experiment_2]

    for i in range(n_replications):
        for j in range(n_stations):
            stations_results_exp_1[j].append(stations_population_data_exp_1[i][j])

    for i in range(n_replications):
        for j in range(n_stations):
            stations_results_exp_2[j].append(stations_population_data_exp_2[i][j])

    for station_id in range(n_stations):
        for i in range(n_replications):
            occupation_values_per_station_along_runs[0].append(
                np.array([xy[1] for xy in stations_results_exp_1[station_id][i]['percentageOfOccupation']['plainDataTable'][:-1]])
            )
            occupation_values_per_station_along_runs[1].append(
                np.array([xy[1] for xy in stations_results_exp_2[station_id][i]['percentageOfOccupation']['plainDataTable'][:-1]])
            )
        occupation_stats_exp_1 = pd.DataFrame(occupation_values_per_station_along_runs)
        occupation_statistics_exp_2 = pd.DataFrame(occupation_values_per_station_along_runs)

    # for station_id in range(10):
    #     print(f'-> generating histogram for station {station_id}')
    #     station_name = stations_results[station_id][0]['name']
    #     station_code = stations_results[station_id][0]['stationCode']
    #     occupied_percentage_along_runs = []

    #     for i in range(n_executions):
    #          occupied_percentage_along_runs.append(
    #              np.array([xy[1] for xy in stations_results[station_id][i]['percentageOfOccupation']['plainDataTable'][:-1]])
    #          )

    #     occupied_percentage_along_runs = np.array(occupied_percentage_along_runs, dtype=object).flatten()

        # fig, ax = plt.subplots()
        # ax.hist(occupied_percentage_along_runs, bins=5)
        # ax.set_title(f"Estación {station_name} | ID: {station_code}")
        # ax.set_xlabel('Porcentaje de ocupación (%)')
        # ax.set_ylabel('Count')
        # fig.savefig(f'../Modelo/MiBiciTuBici/processed-model-output/histogram-code{station_code}.pdf', dpi=100)


    print('--> Done. ✅')
    print(f'--> Results available at f{processed_output_path}. ✅')
