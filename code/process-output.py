import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import scipy.stats
import seaborn as sns
import os
sns.set()

# Estructura del json de salida:
#     model-output-replication-1.json
#     ├── totalLostCustomers
#     ├── totalUsers
#     ├── co2Avoided
#     ├── co2KgPerKm
#     ├── estaciones
#     │   ├── estacion 0
#     │   │   ├── bikesDeposited:6
#     │   │   ├── stationCode:1
#     │   │   ├── clientsWaited:1
#     │   │   ├── address:"Oro?o 1658"
#     │   │   ├── _index:0
#     │   │   ├── bikes:1
#     │   │   ├── latitude:0
#     │   │   ├── currentBikes:1
#     │   │   ├── retiredBikes:12
#     │   │   ├── lostTime:15.533333333333719
#     │   │   ├── percentageOfOccupation: [ ... ]
#     │   │   ├── capacity:20
#     │   │   ├── anchor:19
#     │   │   ├── currentAnchors:0
#     │   │   ├── name:"Museo Castagnino"
#     │   │   ├── lostClients:10
#     │   │   └── longitude:0
#     │   ├── estacion 1
#     │   │   ├── bikesDeposited:6
#     │   │   ├── stationCode:2
#     │   │   ├── clientsWaited:1
#     │   │   ├── address: ...
#     │   └── ...
#     ├── idleBikes
#     ├── totalLostTime
#     ├── totalKm
#     └── busyBikes

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

# results_exp_1
# ├── 'totalUsers': [rep1, rep2, rep3,...,rep100]
# ├── 'co2Avoided':[rep1, rep2, rep3,...,rep100]
# ├── 'totalKm': [rep1, rep2, rep3,...,rep100]
# ├── 'totalLostCustomers': [rep1, rep2, rep3,...,rep100]
# ├── 'delayedUsers': [rep1, rep2, rep3,...,rep100]
# └── 'totalLostTime': [rep1, rep2, rep3,...,rep100]

    stations_population_data_exp_1 = []
    stations_population_data_exp_2 = []

# stations_populations_data_exp_1
# ├── replication 1
# │   ├── estacion 1
# │   ├── estacion 2
# │   ├── estacion 3
# │   └── ...
# ├── replication 2
# │   ├── estacion 1
# │   ├── estacion 2
# │   ├── estacion 3
# │   └── ...
# ├── replication 3
# │   ├── estacion 1
# │   ├── estacion 2
# │   ├── estacion 3
# │   └── ...
# ├── ...
# └── replication 100

# Load files content in memory
    for output_file in experiment_1_files:
        with open(f'{model_output_path}/experiment-1/{output_file}') as f:
            data = json.load(f)

        stations_population_data_exp_1.append(data['estaciones'])

        for statistic in statistics:
            results_exp_1[statistic].append(data[statistic])

    for output_file in experiment_2_files:
        with open(f'{model_output_path}/experiment-2/{output_file}') as f:
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

    confidence_intervals = pd.DataFrame(confidence_intervals).transpose()
    confidence_intervals.columns = ('Cota inferior', 'Media', 'Cota superior')
    confidence_intervals = confidence_intervals.round(decimals=2)
    confidence_intervals.to_csv('../Modelo/MiBiciTuBici/processed-model-output/intervals.csv')
    # ------------------------------------------

    # --> Summary of results for all statistics of interest
    print('-> generating summary of results...')
    summary_exp_1 = pd.DataFrame(results_exp_1).describe().round(decimals=2)
    summary_exp_2 = pd.DataFrame(results_exp_2).describe().round(decimals=2)
    summary_exp_1.to_csv(f'{processed_output_path}/exp-1-statistics-summary.csv')
    summary_exp_2.to_csv(f'{processed_output_path}/exp-2-statistics-summary.csv')
    # -----------------------------------------------------

    # Per-station analysis
    print('-> per-station statistics')
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

    station_names = [stations_results_exp_1[i][0]['name'] for i in range(n_stations)]
    station_codes = [stations_results_exp_1[i][0]['stationCode'] for i in range(n_stations)]

    print('-> generating utilization summaries...')
    for station_id in range(n_stations):
        for i in range(n_replications):
            occupation_values_per_station_along_runs[0].append(
                np.array([xy[1] for xy in stations_results_exp_1[station_id][i]['percentageOfOccupation']['plainDataTable'][:-1]])
            )
            occupation_values_per_station_along_runs[1].append(
                np.array([xy[1] for xy in stations_results_exp_2[station_id][i]['percentageOfOccupation']['plainDataTable'][:-1]])
            )
        occupation_values_exp_1 = pd.DataFrame(occupation_values_per_station_along_runs[0]).transpose()
        occupation_values_exp_2 = pd.DataFrame(occupation_values_per_station_along_runs[1]).transpose()

        summary_occupation_exp_1 = occupation_values_exp_1.describe().round(decimals=2)
        summary_occupation_exp_2 = occupation_values_exp_2.describe().round(decimals=2)
        summary_occupation_exp_1.to_csv(f'{processed_output_path}/exp-1-occupation-summary.csv')
        summary_occupation_exp_2.to_csv(f'{processed_output_path}/exp-2-occupation-summary.csv')

    print('-> generating utilization visualizations...')
    for exp_id in range(1,3):
        fig, ax = plt.subplots(figsize=(25,6))
        ax.bar(x=station_codes, height=eval(f'summary_occupation_exp_{exp_id}').loc['50%', :], color=f'C{exp_id}')
        ax.set_ylabel('Porcentaje de ocupación (%)', fontdict={'size': 17})
        ax.set_xlabel('Código de estación', fontdict={'size': 17})
        ax.set_xticks(station_codes)
        ax.set_title(f'Utilización de las estaciones en el escenario {exp_id}', fontdict={'size': 20})
        ax.set_xlim(0,n_stations+1)
        fig.savefig(f'{processed_output_path}/stations-utilization-{exp_id}.pdf', dpi=100)

    print('-> generating distribution visualization for the stations with highest utilization')
    for exp_id in range(1,3):
        highest_utilization = eval(f'summary_occupation_exp_{exp_id}').loc['50%'].max()
        highest_utilization_id = eval(f'summary_occupation_exp_{exp_id}').loc['50%'].argmax()
        station_name = station_names[highest_utilization_id]

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(eval(f'occupation_values_exp_{exp_id}')[highest_utilization_id], color=f'C{exp_id}')
        ax.set_title(f'Exp. {exp_id}: Distribución de la utilización de la estación {station_name}', fontdict={'size': 14})
        ax.set_xlabel('Porcentaje de utilización (%)', fontdict={'size': 14})
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=len(occupation_values_exp_1)))
        fig.savefig(f'{processed_output_path}/highest-utilization-distribution-{exp_id}.pdf', dpi=100)

    print('--> Done. ✅')
    print(f'--> Results available at {processed_output_path}.')
