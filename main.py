import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

#loads the data from the csv file
def load_data(filename):
    return pd.read_csv(filename)

#make a graph(visual representation) using the given data
def plot_data(data):
    plt.scatter(data['Hours'], data['Score'], color='green')
    plt.title('Hours studied vs Score')
    plt.xlabel('Hours studies')
    plt.ylabel('Score')
    plt.grid(True)
    plt.show()

#making a linear model and training it
def train_model(data):
    x=data[['Hours']]
    y=data['Score']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    model=LinearRegression()
    model.fit(x_train, y_train)
    return model, x_test, y_test

#predicting new scores and finding the mean absolute error
def evaluate_model(model, x_test, y_test):
    y_pred = model.predict(x_test)
    print("\nPredictes vs Actual scores: ")
    for actual, predicted in zip(y_test,y_pred):
        print(f"Actual: {actual:.2f} - Predicted: {predicted:.2f}")
    mea = mean_absolute_error(y_test,y_pred)
    print(f"\nMean Absolute Error: {mea:.2f}")

#input a custom hour and make the model predict the score
def predict_custom(model):
    try:
        hours = float(input("\nEnter the hours studies: "))
        predict_score = model.predict([[hours]])[0]
        print(f"Predicted score: {predict_score:.2f}")
    except ValueError:
        print("Please enter a valid number!")

data = load_data("data.csv")
plot_data(data)
model, x_test, y_test =  train_model(data)
evaluate_model(model, x_test, y_test)
predict_custom(model)