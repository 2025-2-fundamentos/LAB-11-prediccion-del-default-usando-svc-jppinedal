# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
# - Renombre la columna "default payment next month" a "default"
# - Remueva la columna "ID".
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Descompone la matriz de entrada usando PCA. El PCA usa todas las componentes.
# - Estandariza la matriz de entrada.
# - Selecciona las K columnas mas relevantes de la matrix de entrada.
# - Ajusta una maquina de vectores de soporte (svm).
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#
import os
import json
import gzip
import pickle
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import make_scorer

def pregunta_1():
    def load_data(path: str) -> pd.DataFrame:
        return pd.read_csv(path, index_col=False, compression="zip")

    def clean_data(df: pd.DataFrame) -> pd.DataFrame:
        df = df.rename(columns={"default payment next month": "default"})
        df.drop(columns=["ID"], errors="ignore", inplace=True)

        df = df[df["MARRIAGE"] != 0]
        df = df[df["EDUCATION"] != 0]
        df["EDUCATION"] = df["EDUCATION"].apply(lambda x: x if x in [1, 2, 3] else 4)

        return df.dropna()

    # ------------------------------
    #   PIPELINE Y GRID SEARCH
    # ------------------------------
    def make_pipeline(x_train: pd.DataFrame) -> Pipeline:
        cat_cols = ["SEX", "EDUCATION", "MARRIAGE"]
        num_cols = [c for c in x_train.columns if c not in cat_cols]

        preprocessor = ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
                ("scaler", StandardScaler(), num_cols),
            ],
            remainder="passthrough",
        )

        return Pipeline(
            steps=[
                ("prep", preprocessor),
                ("pca", PCA()),
                ("select", SelectKBest(score_func=f_classif)),
                ("clf", SVC(kernel="rbf", random_state=12345, max_iter=-1)),
            ]
        )

    def optimize_pipeline(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series) -> GridSearchCV:
        param_grid = {
            "pca__n_components": [20, X.shape[1] - 2],
            "select__k": [12],
            "clf__kernel": ["rbf"],
            "clf__gamma": [0.1],
        }

        cv = StratifiedKFold(n_splits=10)
        scorer = make_scorer(balanced_accuracy_score)

        gs = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring=scorer,
            cv=cv,
            n_jobs=-1,
        )

        gs.fit(X, y)
        return gs

    # ------------------------------
    #       GUARDADO MODELO
    # ------------------------------
    def save_model(model, path="files/models/model.pkl.gz") -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with gzip.open(path, "wb") as f:
            pickle.dump(model, f)

    # ------------------------------
    #         MÉTRICAS
    # ------------------------------
    def pack_metrics(y_true, y_pred, dataset: str) -> dict:
        return {
            "type": "metrics",
            "dataset": dataset,
            "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
            "balanced_accuracy": round(balanced_accuracy_score(y_true, y_pred), 4),
            "recall": round(recall_score(y_true, y_pred), 4),
            "f1_score": round(f1_score(y_true, y_pred), 4),
        }

    def pack_confusion(y_true, y_pred, dataset: str) -> dict:
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        return {
            "type": "cm_matrix",
            "dataset": dataset,
            "true_0": {"predicted_0": int(cm[0][0]), "predicted_1": int(cm[0][1])},
            "true_1": {"predicted_0": int(cm[1][0]), "predicted_1": int(cm[1][1])},
        }

    def save_jsonl(items: list, path="files/output/metrics.json", append=False) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        mode = "a" if append else "w"
        with open(path, mode, encoding="utf-8") as fh:
            for item in items:
                fh.write(json.dumps(item) + "\n")

    # ------------------------------
    #          EJECUCIÓN
    # ------------------------------
    train = clean_data(load_data("files/input/train_data.csv.zip"))
    test = clean_data(load_data("files/input/test_data.csv.zip"))

    X_train, y_train = train.drop(columns=["default"]), train["default"]
    X_test, y_test = test.drop(columns=["default"]), test["default"]

    pipeline = make_pipeline(X_train)
    estimator = optimize_pipeline(pipeline, X_train, y_train)

    save_model(estimator)

    y_train_pred = estimator.best_estimator_.predict(X_train)
    y_test_pred = estimator.best_estimator_.predict(X_test)

    # métricas
    metrics = [
        pack_metrics(y_train, y_train_pred, "train"),
        pack_metrics(y_test, y_test_pred, "test"),
    ]
    save_jsonl(metrics)

    # matrices
    cms = [
        pack_confusion(y_train, y_train_pred, "train"),
        pack_confusion(y_test, y_test_pred, "test"),
    ]
    save_jsonl(cms, append=True)


if __name__ == "__main__":
    pregunta_1()