import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
import streamlit as st



st.title("Iris Flower Classifier")
st.write("This is a web app that uses a machine learning model to predict the type of Iris flower type.")

st.sidebar.header("User Input Paramaters:")

def user_input_features():
    sepal_length = st.sidebar.slider("Sepal Length", 4.3, 7.9, 5.4)
    sepal_width = st.sidebar.slider("Sepal Width", 2.0, 4.4, 3.4)
    petal_length = st.sidebar.slider('Petal length', 1.0, 6.9, 1.3)
    petal_width = st.sidebar.slider('Petal width', 0.1, 2.5, 0.2)

    data = {
        "Sepal Length" : sepal_length,
        "Sepal Width" : sepal_width,
        "Petal Length" : petal_length,
        "Petal Width" : petal_width
    }

    return (pd.DataFrame(data, index = [0]))

df = user_input_features()

st.subheader("User Input Paramters:")
st.write(df)

iris_dataset = datasets.load_iris()
X = iris_dataset.data
Y = iris_dataset.target

clf = RandomForestClassifier()
tree = clf.fit(X, Y)

pred = clf.predict(df)
pred_prob = clf.predict_proba(df)

st.subheader("Class Lables + Corresponding Index:")
st.write(iris_dataset.target_names)

st.subheader("Prediction:")
st.write(iris_dataset.target_names[pred])

st.subheader("Prediction Probability:")
st.write(pred_prob)

