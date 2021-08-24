import pandas as pd
import json
import requests

if __name__ == "__main__":
    response = requests.get('https://www.mibicitubici.gob.ar/v1/stations')
    if response.status_code != requests.codes.ok:
        response.raise_for_status()

    estaciones = json.loads(response.text)['data']['stations']

    df = pd.DataFrame(estaciones)
    df.drop('location', axis='columns', inplace=True)

    latitudes = [float(estacion['location']['latitude']) for estacion in estaciones]
    longitudes = [float(estacion['location']['longitude']) for estacion in estaciones]
    df['latitude'] = latitudes
    df['longitude'] = longitudes

    df.to_excel('../../datos/geo/estaciones.xlsx', index=None, header=True)
