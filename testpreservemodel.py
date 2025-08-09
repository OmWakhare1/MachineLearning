from pathlib import Path
import joblib
import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix



##CONFIGURATRION DETAILS
ARTIFACTS = Path("artifacts_sample")
ARTIFACTS.mkdir(exist_ok = True)
MODEL_PATH = ARTIFACTS / "diabetes_pipeline.joblib"
RANDOM_STATE = 42
TEST_SIZE = 0.2

def main():

    Labels = ['Negative' , 'Positive'] 

    pipe = joblib.load(MODEL_PATH)

    sample = np.array([[6,148,72,35,0,33.6,0.627,50]])

    Y_pred = pipe.predict(sample)[0]

    print("Predicted result is :",Labels[Y_pred])



if __name__ == "__main__":
    main()



