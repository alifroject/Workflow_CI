import gdown
import dagshub
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import os

dagshub.init(repo_owner='alifroject',
             repo_name='Eksperimen_SML_Alif-Jovani-Safik',
             mlflow=True)

# Load dataset lokal
data = pd.read_csv("MLProject/kc_house_preprocessed.csv")
X = data.drop(columns=["price"])
y = data["price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

param_grid = {"n_estimators": [100, 200], "max_depth": [10, 20]}
grid = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=3, scoring="r2")

with mlflow.start_run(run_name="RandomForest_Advanced"):
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    mlflow.log_params(grid.best_params_)
    mlflow.log_metric("r2", r2)
    mlflow.log_metric("MAE", mae)
    mlflow.log_metric("MSE", mse)

    artifacts_path = "artifacts/model"
    os.makedirs(artifacts_path, exist_ok=True)
    mlflow.sklearn.save_model(best_model, path=artifacts_path)

  
    model_file_id = '1nsiwAD8TmsUfbiiAwYiPh-BH6RvADEwq' 
    model_url = f'https://drive.google.com/uc?id={model_file_id}'
    gdown.download(model_url, os.path.join(artifacts_path, 'model.pkl'), quiet=False)
