import json
import requests
from cargando_dataset import load_dataset

port = 8888

if __name__ == '__main__':
    # Cargando dataset
    x_train, x_test, y_train, y_test = load_dataset(
        'TotalesNacionales.csv', ','
    )
    # Prediciendo los primeros 10 datos
    input_data = x_test[:10].to_json(orient="split")
    endpoint = "http://localhost:{}/invocations".format(port)
    headers = {"Content-type": "application/json; format=pandas-split"}
    prediction = requests.post(endpoint, json=json.loads(input_data), headers=headers)
    print(prediction.text)
