from sklearn import tree
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import accuracy_score
import time

def TrainTest(data):
    """Devide the data in train and test datasets"""
    y = data.iloc[:, -1]
    data_train, data_test, y_train, y_test = model_selection.train_test_split(
        # DONE
        data, y, test_size=0.30, random_state=42)
    return data_train, data_test, y_train, y_test

def main():
    data_types = {
        "citric acid": float,
        "residual sugar": float,
        "sugar": float,
        "sulphates": float,
        "alcohol": float,
        "type": int
    }
    columns = ["citric acid", "residual sugar", "pH", "sulphates", "alcohol", "type"]
    data = pd.read_csv("wine_dataset.csv", header=0, dtype=data_types, names=columns)
    data_train, data_test, y_train, y_test = TrainTest(data)
    data_train = data_train.drop('type', axis=1)
    data_test = data_test.drop('type',axis=1)

    clf = tree.DecisionTreeClassifier(random_state=0, max_depth=10, min_samples_split=3, criterion="gini")
    clf = clf.fit(data.drop('type', axis=1), data['type'])  # Exclude 'type' column for features

    # Print the maximum depth of the decision trees
    max_depth = clf.tree_.max_depth
    print("Maximum Depth of the Decision Tree:", max_depth)

    accuracy_on_training_data = []
    accuracy_on_test_data = []
    training_times = []

    # Accuracy on the training data
    max_depth = 18
    for i in range(1, max_depth + 1):
        print(f"========{i} th iteration========")
        start_time = time.time()  # Record start time
        clf = tree.DecisionTreeClassifier(random_state=0, max_depth=i, min_samples_split=3)
        clf = clf.fit(data.drop('type', axis=1), data['type'])  # Exclude 'type' column for features
        """
        # Compute the accuracy :
        result_on_training_data = clf.predict(data_train)
        accuracy_on_training_data.append(accuracy_score(y_train, result_on_training_data))

        result_on_test_data = clf.predict(data_test)
        accuracy_on_test_data.append(accuracy_score(y_test, result_on_test_data))
        """
        end_time = time.time()  # Record end time
        computation_time = end_time - start_time
        training_times.append(computation_time)

    x = list(range(1, max_depth + 1))

    plt.figure(figsize=(8, 4))
    ax = plt.gca()

    # ax.plot(x, accuracy_on_training_data, label='Train', color='blue')
    # ax.plot(x, accuracy_on_test_data, label='Test', color='red')
    ax.plot(x, training_times, label='Train', color='blue')

    #ax.set_xlabel('max-depth')
    #ax.set_ylabel('Accuracy')
    #ax.set_title('Decision tree accuracy')
    ax.set_title('Decision tree time to compute')
    ax.set_ylabel('time')
    ax.legend()

    ax.set_xticks(range(1, max_depth + 1))

    plt.show()
"""
    # Accuracy :
    result_on_training_data = clf.predict(data_train)
    print(f"Accuracy score on training dataset: {accuracy_score(y_train, result_on_training_data)}")

    result_on_test_data = clf.predict(data_test)
    print(f"Accuracy score on test dataset: {accuracy_score(y_test, result_on_test_data)}")
"""

main()
