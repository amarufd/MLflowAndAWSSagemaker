import json, boto3

def lambda_handler(event, context):
    
    app_name = 'covid-19-lr-demo'
   
    body = event['body']
    print('body==>', body)
    
    sm = boto3.client('runtime.sagemaker')
    
    prediction = sm.invoke_endpoint(
        EndpointName=app_name,
        Body=body,
        ContentType='application/json; format=pandas-split'
    )
    prediction = prediction['Body'].read().decode("ascii")
    
    return {
        'statusCode': 200,
        'headers' : {
            'Access-Control-Allow-Origin' : '*'
        },
        'body': prediction
    }