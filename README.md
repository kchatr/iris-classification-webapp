# Iris Classifier Web App

This web app allows the users to adjust the parameters and see how the model dynamically predicts what kind of Iris it would be, along with the probability that it fits into the class. It is intended to showcase the uses of decision trees and random forests, as well as how they work and adapt under the hood.

This model uses a random forrest classifier (to reduce overfitting and increase accuracy from a standard decision tree classifier), imported from Scikit-Learn. It then uses Streamlit to convert the model into an interactive web app.

A tree.dot file has been generated that shows the users how the model works, and the inner workings of a decision tree.
