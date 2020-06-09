import boto3
from load_dataset import load_dataset

app_name = 'covid-19-lr-demo'
region = 'us-east-X'

if __name__ == '__main__':
    sm = boto3.client('sagemaker', region_name=region)
    smrt = boto3.client('runtime.sagemaker', region_name=region)

    # Revisando status de Endpoints
    endpoint = sm.describe_endpoint(EndpointName=app_name)
    print("Endpoint status: ", endpoint["EndpointStatus"])
    # Cargando dataset
    x_train, x_test, y_train, y_test = load_dataset(
        'datos/TotalesNacionales.csv', ','
    )
    # Prediciendo los primeros 10 datos
    input_data = x_test[:10].to_json(orient="split")
    prediction = smrt.invoke_endpoint(
        EndpointName=app_name,
        Body=input_data,
        ContentType='application/json; format=pandas-split'
    )
    prediction = prediction['Body'].read().decode("ascii")
    print(prediction)
