import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_graphviz
import streamlit as st

from PIL import Image

st.title("Iris Flower Classifier")
st.write("This is a web app that uses a random forest clasifier to predict the type of an Iris flower.")
st.write("Random forest classifiers use multiple decision trees to come up with a model. Each tree outputs a prediction, and the tree with the highest accurate becomes the one our model chooses.")

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

st.subheader("Current Input Paramters:")
st.write(df)

iris_dataset = datasets.load_iris()
X = iris_dataset.data
Y = iris_dataset.target

clf = RandomForestClassifier()
forest = clf.fit(X, Y)

pred = clf.predict(df)
pred_prob = clf.predict_proba(df)

st.subheader("Class Lables and their Corresponding Index:")
st.write(iris_dataset.target_names)

st.subheader("__Prediction:__")
st.write(str(iris_dataset.target_names[pred])[2:-2].capitalize())

st.subheader("Prediction Probability:")
st.write(pred_prob)

decision_tree_vis = Image.open("decision_tree.png")
st.image(decision_tree_vis, caption = "The Decision Tree Used by the Classification Model", use_column_width = True)

st.write("Here's the decision tree that the model created based on the training data. Go through the tree, and see how your parameters correspond to the prediction!")

