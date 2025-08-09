# 🩺 Diabetes Predictor Application

This is a **Python-based Machine Learning application** that predicts the likelihood of diabetes using the **Pima Indians Diabetes Dataset** (or any similar dataset with the same structure).

It uses:
- **Logistic Regression** for classification
- **Correlation-based feature importance visualization**
- **Model preservation** using `joblib`

---

## Features
- Load and explore a CSV dataset
- Display dataset dimensions and statistical summary
- Visualize feature importance using:
  - Heatmap (Seaborn)
  - Correlation bar chart
- Split data into training and testing sets
- Train a Logistic Regression model
- Display model accuracy
- Save (preserve) the trained model to disk

---

##  Project Structure
```
DiabetesCaseHomeWork/
│── Diabetespredictor.py     # Main application script
│── requirements.txt         # Python dependencies
│── README.md                # Project documentation
│── artifacts_sample/        # Folder to store trained models
│   └── diabetes_pipeline.joblib
│── diabetes.csv             # Example dataset (Pima Indians)
```

---

## ⚙️ Installation
### 1. Clone or Download the Project
```bash
git clone https://github.com/yourusername/DiabetesPredictor.git
cd DiabetesPredictor
```

### 2. Create Virtual Environment (Recommended)
```bash
python -m venv venv
source venv/bin/activate   # On Linux/Mac
venv\Scripts\activate      # On Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## How to Run
1. Place your dataset CSV file (e.g., `diabetes.csv`) in the **same directory** as `Diabetespredictor.py`.
2. Run the application:
```bash
python Diabetespredictor.py
```
3. When prompted, enter the CSV file name (e.g., `diabetes.csv`).
4. The application will:
   - Display dataset info
   - Show feature importance visualizations
   - Train a Logistic Regression model
   - Show accuracy score
   - Save the trained model in `artifacts_sample/`

---

## Example Dataset Format
The dataset must have the following columns:
```
Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age,Outcome
6,148,72,35,0,33.6,0.627,50,1
```
- `Outcome = 1` → Positive (Diabetic)
- `Outcome = 0` → Negative (Non-Diabetic)

---

## Example Output

**Feature Importance Heatmap**  
The program will display a heatmap of feature correlations:  

![Heatmap Example](https://via.placeholder.com/600x350?text=Feature+Importance+Heatmap)

**Bar Chart of Correlation with Target**  
A bar chart showing correlation of each feature with the target variable (`Outcome`):  

![Bar Chart Example](https://via.placeholder.com/600x350?text=Feature+Importance+Bar+Chart)


---

##  Model Saving
The trained model is saved in:
```
artifacts_sample/diabetes_pipeline.joblib
```
You can load it later with:
```python
import joblib
model = joblib.load("artifacts_sample/diabetes_pipeline.joblib")
```

---

## 📦 Requirements
- Python 3.10+  
- See `requirements.txt` for dependencies:
  ```
  pandas
  matplotlib
  seaborn
  scikit-learn
  joblib
  ```

---

## 🧠 Author
**Om Ravindra Wakhare**  
📅 *Created on:* 8th September 2025

---
