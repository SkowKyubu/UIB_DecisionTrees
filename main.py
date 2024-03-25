import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import time


class TreeNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, info_gain=None, value=None):
        """constructor"""
        self.feature = feature  # Selected feature for the split
        self.threshold = threshold  # threshold value to split the feature
        self.left = left  # Left subtree (lower or equal values)
        self.right = right  # Right subtree (upper or equal values)
        self.info_gain = info_gain  # Information gain of the split
        self.value = value  # Value for the leaf (for leaf nodes)


class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=2):
        """constructor"""
        self.root = None  # Root node of the tree
        self.max_depth = max_depth-1  # Stopping condition
        self.min_samples_split = min_samples_split
        self.function = self.entropy

    def learn(self, data, impurity_measure='entropy', prune=False):
        """function to create the tree based on data"""
        if impurity_measure == 'gini':
            self.function = self.gini
        if prune is True:
            data_train, data_prune, y_train, y_prune = trainTest(data)
            self.root = self.build_tree(data_train)
            accuracy = accuracy_score(y_prune, self.prediction(data_prune))
            self.root = self.prune(self.root, data=data_train, y=y_train,
                                   data_prune=data_prune, y_prune=y_prune, accuracy=accuracy)
        else:
            self.root = self.build_tree(data)
        return self

    def build_tree(self, data, current_depth=0, depth=[]):
        """recursive function to build the tree"""
        # Print the current_depth
        if current_depth not in depth:
            depth.append(current_depth)
            print(current_depth)
        #  If all data points have the same label :
        y = data.iloc[:, -1]
        if y.nunique() == 1:
            leaf_value = self.calculate_leaf_value(y)
            #  return a leaf with that label
            return TreeNode(value=leaf_value)

        # Else, if all data points have identical feature values:
        all_rows_same = data.duplicated().sum() == 0
        if all_rows_same is True:
            leaf_value = self.calculate_leaf_value(y)
            return TreeNode(value=leaf_value)

        # Else, choose a feature that maximizes the information gain :
        num_samples, num_features = data.shape
        if num_samples >= self.min_samples_split and current_depth <= self.max_depth:
            best_split_label, best_split_value, best_split_ig = self.get_best_split(y, data)
            # We check if the split is pure
            if best_split_ig is not None and best_split_ig > 0:
                data_left, data_right = self.make_split(best_split_label, best_split_value, data)
                left_subtree = self.build_tree(data_left, current_depth+1, depth)
                right_subtree = self.build_tree(data_right, current_depth + 1, depth)
                return TreeNode(best_split_label, best_split_value, left_subtree, right_subtree, best_split_ig)
            else:
                leaf_value = self.calculate_leaf_value(y)
                return TreeNode(value=leaf_value)
            # compute leaf nodel
        leaf_value = self.calculate_leaf_value(y)
        # return leaf node
        return TreeNode(value=leaf_value)

    def print_tree(self, tree=None, indent="", max_depth=10):
        """function to print the tree"""
        if not tree:
            tree = self.root
        if tree.value is not None or max_depth == 0:
            print(tree.value)
        else:
            print(f"{str(tree.feature)} <= {tree.threshold} ig : {tree.info_gain}")
            print(f"{indent}left:", end="")
            self.print_tree(tree.left, indent + "          ", max_depth - 1)
            print(f"{indent}right:", end="")
            self.print_tree(tree.right, indent + "          ", max_depth - 1)

    def prune(self, node, data, y, data_prune, y_prune, accuracy):
        """Prune function to cut the useless branches of the tree"""
        # If it's not a leaf :
        if node.value is None:
            # We save the two subtrees
            node_save_left = node.left
            node_save_right = node.right
            # If the subtree are actually two leafs with the same value :
            if node_save_left.value is not None and node_save_right.value is not None:
                if node_save_left.value == node_save_right.value:
                    # We replace the node with a leaf
                    return TreeNode(value=node.left.value)
            data1 = data[data[node.feature] < node.threshold]
            data2 = data[data[node.feature] >= node.threshold]

            y1 = data1.iloc[:, -1]
            y2 = data2.iloc[:, -1]

            value1 = y1.mode().iloc[0]
            value2 = y2.mode().iloc[0]
            # We build to new leaf for the actual node :
            node.left = TreeNode(value=value1)
            node.right = TreeNode(value=value2)

            # Compute the accuracy with this configuration
            result = self.prediction(data_prune)

            # if the new accuracy is not improved :
            if accuracy_score(y_prune, result) < accuracy:
                # We continue to explore the tree with the original node
                node.left = self.prune(node_save_left, data1, y1, data_prune, y_prune, accuracy)
                node.right = self.prune(node_save_right, data2, y2, data_prune, y_prune, accuracy)
            # otherwise we return the new node
            return node
        # if it's a leaf :
        return node

    def gini(self, y):
        """Gini index"""
        p = y.value_counts()/y.shape[0]
        return np.sum(p*(1-p))

    def entropy(self, y):
        """Calculate the entropy of a variable Y"""
        p = y.value_counts() / y.shape[0]
        return -np.sum(p*np.log2(p))

    def information_gain(self, y, mask):
        """Calculate the information gain of a variable Y using a mask"""
        cardinal = mask.shape[0]
        m_true = sum(mask)
        m_false = cardinal - m_true

        if m_true == 0 or m_false == 0:
            ig = 0
        else:
            ig = self.function(y) - m_true/cardinal * self.function(y[mask]) - m_false/cardinal\
                 * self.function(y[-mask])
            # y[mask] and y[-mask] are respectively two vectors containing the values of y where the mask is true and
            # false.
        return ig

    def max_information_gain(self, x, y):
        """Maximise the information gain"""
        ig = 0
        split = None
        values = x.sort_values().unique()
        for split_value in values:
            mask = x < split_value
            ig_loop = self.information_gain(y, mask)
            if ig_loop > ig:
                ig = ig_loop
                split = split_value
        return ig, split

    def get_best_split(self, y, data):
        """Find the best split among the information gain"""
        ig = data.drop('type', axis=1).apply(self.max_information_gain, y=y)
        ig = ig.loc[:, ig.loc[0] > 0]
        if ig.empty:
            # Handle the case where no valid split is found
            return None, None, None
        # Get the results for the split with the highest information gain
        split_variable = ig.iloc[0].astype(np.float32).idxmax()
        split_value = ig.loc[1, split_variable]
        split_ig = ig.loc[0, split_variable]

        return split_variable, split_value, split_ig

    def make_split(self, variable, value, data):
        """Devide the dataset in two part"""
        data1 = data[data[variable] < value]
        data2 = data[data[variable] >= value]
        return data1, data2

    def calculate_leaf_value(self, y):
        """Calculate the most present value in the leaf"""
        return y.mode().iloc[0]

    def prediction(self, x):
        """Make predictions on a set of data"""
        result = []
        for index, row in x.iterrows():
            pred = self.make_prediction(row, self.root)
            result.append(pred)
        return result

    def make_prediction(self, x, node):
        """Function to make a prediction of a single data"""
        if node.value is not None:
            return node.value  # if we reach a leaf
        feature_val = x[node.feature]
        if feature_val <= node.threshold:
            return self.make_prediction(x, node.left)
        else:
            return self.make_prediction(x, node.right)


def trainTest(data):
    """Devide the data in train and test datasets"""
    y = data.iloc[:, -1]
    data_train, data_test, y_train, y_test = model_selection.train_test_split(
        data, y, test_size=0.30, random_state=42)
    return data_train, data_test, y_train, y_test


def load_data():
    # Load the dataset :
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
    return data


def print_result(max_depth, measure='accuracy', impurity_measure='entropy', print_tree=False, prune=False):
    data = load_data()
    data_train, data_test, y_train, y_test = trainTest(data)
    tree = DecisionTree(min_samples_split=3, max_depth=max_depth)
    if measure == 'time':
        start_time = time.time()  # Record start time
    tree = tree.learn(data_train, impurity_measure=impurity_measure, prune=prune)
    if measure == 'time':
        # Compute the time
        end_time = time.time()  # Record end time
        computation_time = end_time - start_time
    if print_tree is True:
        tree.print_tree(max_depth=tree.max_depth)
    if measure == 'accuracy':
        result_on_training_data = tree.prediction(data_train)
        print(f"Accuracy score on training dataset: {accuracy_score(y_train, result_on_training_data)}")

        result_on_test_data = tree.prediction(data_test)
        print(f"Accuracy score on test dataset: {accuracy_score(y_test, result_on_test_data)}")
    elif measure == 'time':
        print(f"Time to compute: {computation_time}s")


def plot_result(max_depth, measure='accuracy'):
    # Plot graphs with the results
    data = load_data()
    data_train, data_test, y_train, y_test = trainTest(data)

    accuracy_on_training_data = []
    accuracy_on_test_data = []
    training_times = []

    # Accuracy on the training data
    for i in range(1, max_depth+1):
        print(f"========{i} th iteration========")
        if measure == 'time':
            start_time = time.time()  # Record start time

        tree = DecisionTree(min_samples_split=3, max_depth=i)
        tree.root = tree.build_tree(data, depth=[])

        if measure == 'accuracy':
            # Compute the accuracy :
            result_on_training_data = tree.prediction(data_train)
            accuracy_on_training_data.append(accuracy_score(y_train, result_on_training_data))

            result_on_test_data = tree.prediction(data_test)
            accuracy_on_test_data.append(accuracy_score(y_test, result_on_test_data))

        if measure == 'time':
            # Compute the time
            end_time = time.time()  # Record end time
            computation_time = end_time - start_time
            training_times.append(computation_time)

    x = list(range(1, max_depth+1))
    plt.figure(figsize=(8, 4))
    ax = plt.gca()

    if measure == 'accuracy':
        ax.plot(x, accuracy_on_training_data, label='Train', color='blue')
        ax.plot(x, accuracy_on_test_data, label='Test', color='red')
        ax.set_ylabel('Accuracy')
        ax.set_title('Decision tree accuracy')
    elif measure == 'time':
        ax.plot(x, training_times, label='Train', color='blue')
        ax.set_ylabel('time')
        ax.set_title('Decision tree time compute')

    ax.set_xlabel('max-depth')
    ax.legend()
    ax.set_xticks(range(1, max_depth + 1))

    plt.show()


def main():
    # plot_result(3, 'time')
    print_result(5, 'time', 'gini', print_tree=True, prune=False)


main()
