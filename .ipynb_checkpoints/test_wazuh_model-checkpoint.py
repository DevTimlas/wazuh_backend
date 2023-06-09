import joblib
import warnings
from typing import List
import numpy as np
from fastapi import FastAPI, Query

warnings.filterwarnings("ignore")

model_path1 = "rf_model.pkl"
model_path2 = "pca_random_forest_model.pkl"

test_data = joblib.load('test_data.pkl')
test_reduced_data = joblib.load('test_reduced_data.pkl')

def make_predict(data, model_path):
    job_model = joblib.load(model_path)
    result = job_model.predict_proba(data.reshape(1, -1))[:, 1][0]
    return result

app = FastAPI()
@app.get("/make_predict")
def read_items(q: List[float]):
    q = np.array(q)
    print(q.shape)
    try:
        res = make_predict(q, model_path2)
    except:
        res = "there's an error"
    return {"res": res}