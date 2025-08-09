import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from pathlib import Path
import joblib







##CONFIGURATRION DETAILS
ARTIFACTS = Path("artifacts_sample")
ARTIFACTS.mkdir(exist_ok = True)
MODEL_PATH = ARTIFACTS / "diabetes_pipeline.joblib"


###############################################################################################################
# Function name :- load_dataset()
# Description of Function:- This function returns the actual dataset, dimesions of dataset, statistical .
# Author:- Om Ravindra Wakhare
# Date :- 8/09/2025
# Time:- 6:34pm
################################################################################################################

def load_dataset(filename):

    #Use to load read the data set using pandas
    df = pd.read_csv(filename)

    #show the dimension of dataset
    shape = df.shape

    #shows the descripton of dataset
    desc = df.describe()
    

    return df,shape,desc


###############################################################################################################
# Function name :- featureImportance()
# Description of Function:- This function returns the correlation matrix and heatmap,bargraph for visualizing the feature importance.
# Author:- Om Ravindra Wakhare
# Date :- 8/09/2025
# Time:- 6:34pm
################################################################################################################

def featureImportance(df):

    #returns the correlation matrix
    corrmartrix = df.corr()

    #Heatmap plotting for feature importance
    plt.figure(figsize=(12,8))
    sns.heatmap(df.corr(),annot = True, cmap = 'coolwarm')
    plt.show()
    
    #target column
    target = 'Outcome'

    #drop target's self-correlation
    correlations = df.corr()[target].drop(target) 

    #Sorting in the ascending order
    correlations = correlations.reindex(correlations.abs().sort_values(ascending=False).index)

    #bar graph plotting
    plt.figure(figsize=(10, 6))
    correlations.plot(kind='bar', color='skyblue', edgecolor='black')
    plt.title('Feature Importance (Correlation with target)')
    plt.ylabel('Correlation Coefficient')
    plt.show()
    
    return corrmartrix


####################################################################################################################################
# Function name :- DataSplit()
# Description of Function:- This function split the data for training and testing.
# Author:- Om Ravindra Wakhare
# Date :- 8/09/2025
# Time:- 6:34pm
####################################################################################################################################
def DataSplit(df):
    
    #Drop the column Outcome
    x = df.drop(columns = ['Outcome'])
    

    #Take only Outcome
    y = df['Outcome']

    #Standard scaler object
    scaler = StandardScaler()
    x_scale = scaler.fit_transform(x)

    #Use to split the data into training and testing
    X_train,X_test,Y_train,Y_test = train_test_split(x_scale,y,test_size=0.2,random_state=42)

    return X_train,X_test,Y_train,Y_test,scaler



####################################################################################################################################
# Function name :- ModelBuilding()
# Description of Function:- This function builds the model to evaluate the accuracy.
# Author:- Om Ravindra Wakhare
# Date :- 8/09/2025
# Time:- 6:34pm
####################################################################################################################################

def ModelBuilding(x_train,x_test,y_train,y_test):

    #Object creation of LogisticRegression
    model = LogisticRegression()

    #Train the model
    model.fit(x_train,y_train)

    #test the model
    y_pred = model.predict(x_test)

    #Accuracy of the model
    accuracy = accuracy_score(y_pred,y_test)

    return accuracy*100,model




####################################################################################################################################
# Function name :- PreserveTheModel()
# Description of Function:- This function is used to preserve the model on the hardisk.
# Author:- Om Ravindra Wakhare
# Date :- 8/09/2025
# Time:- 6:34pm
####################################################################################################################################


def PreservetheModel(model):

   #Dumping of the model on the HDD 
   joblib.dump(model ,MODEL_PATH)

   




####################################################################################################################################
# Function name :- main()
# Description of Function:- This function is entry point function for our application.
# Author:- Om Ravindra Wakhare
# Date :- 8/09/2025
# Time:- 6:34pm
####################################################################################################################################


def main():
    line = "*"*84
    print(line)
    print("---------------------------------Diabestes predictor Application---------------------------------")
    filename = input("Enter the csv file name which you want to choose and make sure it should be in the same directory of file : \n")
    print(line)


    
    data = load_dataset(filename)


    
    print(line)
    print("The first five values of dataset is :")
    print(data[0].head())
    print(line)


    
    print(line)
    print("The dimesion of the data are :")
    print(data[1])
    print(line)


    print(line)
    print("The statistical information about the dataset is : \n")
    print(data[2])
    print(line)


    
    print(line)
    feature = featureImportance(data[0])
    print("The correaltion matrix is : \n")
    print(feature)
    print(line)



    print(line)
    split = DataSplit(data[0])
    print("The X_train value is :",split[0])
    print(line)
    print("The X_test value is :",split[1])
    print(line)
    print("The Y_train value is :",split[2])
    print(line)
    print("The Y_test value is :",split[3])



    Model = ModelBuilding(split[0],split[1],split[2],split[3])
    print(line)
    print("The accuracy score is : \n")
    print(Model[0])
    print(line)

    Preserve = PreservetheModel(Model[1])

    if(MODEL_PATH.exists() == True):
        print(line)
        print("Model is succesfully preserved on the hdd")
        print(line)
        print("---------------------------------------Thank you for using our Application--------------------------------------")
    else:
        print(line)
        print("Model is not preserved on the hdd")
        print(line)
        print("---------------------------------------Thank you for using our Application--------------------------------------")

   






    









if __name__ == "__main__":
    main()