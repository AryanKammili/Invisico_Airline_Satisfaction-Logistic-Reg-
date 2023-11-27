import pandas as pd
import warnings
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning

data = pd.read_csv("Invistico_Airline.csv")

data = data[["satisfaction", "Customer Type", "Age", "Class", "Seat comfort"]]

# This allows us to not have to change our response variable every single time #
# A response variable is pretty much a variable that responds to input, so our output #
predict = "satisfaction"

response = data[predict]

# Explanatory variables are pretty much variables that you feed in to get an output, so our input #
explanatory = data.drop(predict, axis=1)

# Turn any string data into numbers #
labelEncoder = LabelEncoder()

satisfaction = labelEncoder.fit_transform(list(data["satisfaction"]))
customerType = labelEncoder.fit_transform(list(data["Customer Type"]))
age = labelEncoder.fit_transform(list(data["Age"]))
cls = labelEncoder.fit_transform(list(data["Class"]))
seat_comfort = labelEncoder.fit_transform(list(data["Seat comfort"]))

x = list(zip(customerType, age, cls, seat_comfort))
y = list(satisfaction)

typesOfSatisfaction = ["satisfied", "dissatisfied"]

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

regression = LogisticRegression()

# Fitting the regression #
regression.fit(x_train, y_train)


y = regression.predict(x_test)

predictions = regression.predict(x_test)

actualSatisfaction = ""
predictedSatisfaction = ""

timesWrong = 0

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=ConvergenceWarning)

for i in range(len(predictions)):

    if i > 5:
        continue
    else:
        actualSatisfaction = typesOfSatisfaction[y_test[i]]
        predictedSatisfaction = typesOfSatisfaction[predictions[i]]
        print("Actual Satisfaction: ", actualSatisfaction, ", Predicted Satisfaction: ", predictedSatisfaction)

    if not actualSatisfaction == predictedSatisfaction:
        timesWrong += 1


print("The model was wrong", timesWrong, "times out of", len(predictions), "predictions")


