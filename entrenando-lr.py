import mlflow.sklearn
from sklearn import linear_model
from sklearn.metrics import accuracy_score,mean_squared_error, r2_score

from cargando_dataset import load_dataset

if __name__ == '__main__':
    mlflow.set_experiment('covid-19-lr')
    with mlflow.start_run(run_name='covid-19-lr-basic') as run:
        x_train, x_test, y_train, y_test = load_dataset(
            'TotalesNacionales.csv', ','
        )

        linear_regr = linear_model.LinearRegression()
        linear_regr.fit(x_train,y_train)
        y_pred = linear_regr.predict(x_test)

        mse = mean_squared_error(y_test,y_pred)
        print("Mean squared error: %.2f" % mse)
        r2 = r2_score(y_test, y_pred)
        print('R2 score: %.2f' % r2 )

        mlflow.log_metric("mse", mse)
        mlflow.log_metric("r2", r2)

        mlflow.sklearn.log_model(linear_regr, "covid-19-lr-model")
        mlflow.end_run()
