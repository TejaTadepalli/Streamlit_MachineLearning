import streamlit as st
from sklearn import datasets
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

st.title("Basic Streamlit Project")
st.write("Using Machine Learning Classifiers on sklearn Datasets")

datagroup_name = st.sidebar.selectbox("Select Dataset: ", ("Iris","Breast Cancer","Wine Dataset"))
st.write("Dataset: ",datagroup_name)

classifiergroup_name = st.sidebar.selectbox("Select Classifier: ",("KNN","SVM","RandomForest"))

# Function to pick the dataset and get its respective information
def pick_dataset(datagroup_name):
    if(datagroup_name == "Iris"):
        data = datasets.load_iris()
    elif(datagroup_name == "Breast Cancer"):
        data = datasets.load_breast_cancer()
    elif(datagroup_name == "Wine Dataset"):
        data = datasets.load_wine()

# In datasets from sklearn, we have "data" which we build our model upon and final "target" which we aim to find the classification for

    X = data.data
    Y = data.target
    
    return X,Y

def put_parameter_ui(classifiergroup_name):
    params = dict()
    if(classifiergroup_name == "KNN"):
        K = st.sidebar.slider("K",1,14)
        params["K"] = K
        return params
    elif(classifiergroup_name == "SVM"):
        C = st.sidebar.slider("C",0.03,9.0)
        params["C"] = C
        return params
    elif(classifiergroup_name == "RandomForest"):
        max_depth_val = st.sidebar.slider("max_depth_val",1,13)
        n_estimate = st.sidebar.slider("n_estimate",1,100)
        params["max_depth_val"] = max_depth_val
        params["n_estimate"] = n_estimate
        return params

def have_classifier(classifiergroup_name,params):
    if(classifiergroup_name == "KNN"):
        classifier = KNeighborsClassifier(n_neighbors=params["K"])
    elif(classifiergroup_name == "SVM"):
        classifier = SVC(C=params["C"])
    elif(classifiergroup_name == "RandomForest"):
        classifier = RandomForestClassifier(n_estimators=params["n_estimate"],max_depth=params["max_depth_val"],random_state=1234)
    return classifier

X,Y = pick_dataset(datagroup_name)
params = put_parameter_ui(classifiergroup_name)
classifier = have_classifier(classifiergroup_name,params)
st.write("Shape of Dataset: ", X.shape)
st.write("Number of Classes: ", len(np.unique(Y)))

# Model Training
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2,random_state=1234)
classifier.fit(X_train,Y_train)
Y_pred = classifier.predict(X_test)

# Metrics
accuracy = accuracy_score(Y_test,Y_pred)
st.write(f"Classifier: {classifiergroup_name}")
st.write(f"Accuracy: {accuracy}")

# Plot for Data
pca = PCA(2)
X_projection = pca.fit_transform(X)
x1 = X_projection[:,0]
x2 = X_projection[:,1]

fig = plt.figure()
plt.scatter(x1,x2,c=Y,alpha=0.8,cmap="viridis")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar()
st.pyplot(fig)