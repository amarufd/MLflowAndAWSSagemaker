import mlflow
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split


def load_dataset_to_mlflow(path, sep, test_size=0.2, random_state=123):
    # carga el dataset
    data = pd.read_csv(path, sep=sep)

    # se procesa el dataset
    x,y = procesando_dataset(data)
    
    # almacena caracteristicas del dataset en MLflow
    mlflow.log_param("dataset_path", path)
    mlflow.log_param("dataset_shape", data.shape)
    mlflow.log_param("test_size", test_size)
    mlflow.log_param("random_state", random_state)
    mlflow.log_param("one_hot_encoding", True)
    return train_test_split(x, y, test_size=test_size, random_state=random_state)

def load_dataset(path, sep, test_size=0.2, random_state=123):
    # carga el dataset
    data = pd.read_csv(path, sep=sep)

    # se procesa el dataset
    x,y = procesando_dataset(data)

    return train_test_split(x, y, test_size=test_size, random_state=random_state)

def procesando_dataset(datos):
    # Transformar string a date para hacer resta
    X = datos.columns[1:].to_series().apply(lambda x: datetime.strptime(x, "%Y-%m-%d"))
    # Restar a todos los días la fecha inicial para obtener el transcurso de dias
    X = X - X[0]
    # Obtener los dias de diferencia
    X = X.apply(lambda x: x.days)
    X = X.to_frame()
    X.columns = ["dias_transcurridos"]
    X["Dias2"] = X.apply(lambda x: x**2)
    
    Y_aux = datos.iloc[1, 1:]
    y = []
    for i in range(0, len(Y_aux)):
        y.append(Y_aux[i])
    y = pd.DataFrame(y)
    y.columns = ["total_contagiados"]
    y.index = X.index

    return X,y

if __name__ == '__main__':
    x_train, x_test, y_train, y_test = load_dataset_to_mlflow(
        'datos/TotalesNacionales.csv', ','
    )
    print(x_train.head())
    print(y_train.head())
