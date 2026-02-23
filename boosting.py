import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from sklearn.ensemble import GradientBoostingClassifier # for comparison

import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

from sklearn.metrics import f1_score # for optimizing threshold

from sklearn.metrics import roc_curve, auc

class DecisionNode:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
        self.feature_index = feature_index  # which column to split on
        self.threshold = threshold
        self.left = left  # left child
        self.right = right  # right child
        self.value = value  # prediction value, only if this is a leaf

class DecisionTree:
    def __init__(self, max_depth=3, min_samples_split=5):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None
        self.feature_counts = {}

    # builds tree by finding best split recursively
    def fit(self, X, y):
        self.root = self.build_tree(X, y, depth=0)

    def build_tree(self, X, y, depth):
        num_samples, num_features = X.shape

        # stop conditions:
        # reached max depth
        # too few samples to split
        # all y values are the same
        if depth >= self.max_depth or num_samples < self.min_samples_split or len(np.unique(y)) == 1:
            leaf_value = np.mean(y)
            return DecisionNode(value=leaf_value)

        # find best split
        best_split = self.get_best_split(X, y, num_features)

        # if gain is 0 return a leaf
        if best_split["var_reduction"] == 0:
            leaf_value = np.mean(y)
            return DecisionNode(value=leaf_value)

        # recording this feature is useful
        chosen_feature = best_split["feature_index"]
        if chosen_feature in self.feature_counts:
            self.feature_counts[chosen_feature] += 1
        else:
            self.feature_counts[chosen_feature] = 1

        # recursion
        left_subtree = self.build_tree(best_split["X_left"], best_split["y_left"], depth + 1)
        right_subtree = self.build_tree(best_split["X_right"], best_split["y_right"], depth + 1)

        return DecisionNode(
            feature_index=best_split["feature_index"],
            threshold=best_split["threshold"],
            left=left_subtree,
            right=right_subtree
        )

    # loops through all features to find the best split
    def get_best_split(self, X, y, num_features):
        best_split = {"var_reduction": 0, "feature_index": None, "threshold": None}
        current_uncertainty = np.var(y) * len(y)  # total variance

        # loop every feature (randomly selecting subset for speed if needed)
        for feature_index in range(num_features):
            values = X[:, feature_index]
            unique_values = np.unique(values)

            # if too many unique values, sampling them to speed up
            if len(unique_values) > 20:
                unique_values = np.percentile(unique_values, [10, 20, 30, 40, 50, 60, 70, 80, 90])

            for threshold in unique_values:
                # sppliting data
                left_mask = values <= threshold
                right_mask = values > threshold

                if sum(left_mask) == 0 or sum(right_mask) == 0:
                    continue

                y_left, y_right = y[left_mask], y[right_mask]

                # calculating the cost (variance reduction)
                current_var_cost = (len(y_left) * np.var(y_left) + len(y_right) * np.var(y_right))
                var_reduction = current_uncertainty - current_var_cost

                if var_reduction > best_split["var_reduction"]:
                    best_split = {
                        "feature_index": feature_index,
                        "threshold": threshold,
                        "var_reduction": var_reduction,
                        "X_left": X[left_mask], "y_left": y_left,
                        "X_right": X[right_mask], "y_right": y_right
                    }
        return best_split

    # traversing the tree for every row in x
    def predict(self, X):
        return np.array([self.make_prediction(x, self.root) for x in X])

    def make_prediction(self, x, node):
        if node.value is not None: return node.value  # if Leaf
        if x[node.feature_index] <= node.threshold:
            return self.make_prediction(x, node.left)
        else:
            return self.make_prediction(x, node.right)


class CustomGradientBoosting:
    def __init__(self, n_estimators=10, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees = []
        self.base_pred = 0

    def fit(self, X, y):
        # initial prediction mean
        self.base_pred = np.mean(y)
        current_predictions = np.full(len(y), self.base_pred)

        print(f"Starting training with base prediction: {self.base_pred:.4f}")

        # boosting loop
        for i in range(self.n_estimators):
            # calculate residuals target - prediction
            residuals = y - current_predictions

            # subsampling random 70% of indices
            n_samples = len(X)
            subsample_size = int(n_samples * 0.7)
            idx = np.random.choice(np.arange(n_samples), subsample_size, replace=False)
            X_subset = X[idx]
            res_subset = residuals[idx]

            # fit a weak learner to the residuals (not the target)
            tree = DecisionTree(max_depth=self.max_depth)
            tree.fit(X_subset, res_subset)

            # updating predictions
            # new prediction = old prediction + (learning rate * tree prediction)
            tree_pred = tree.predict(X)
            current_predictions += self.learning_rate * tree_pred

            # saving the tree
            self.trees.append(tree)

            mse = np.mean(residuals ** 2)
            print(f"Tree {i + 1}/{self.n_estimators} built. MSE: {mse:.5f}")

    def predict(self, X):
        # starting with the base mean
        final_predictions = np.full(len(X), self.base_pred)

        # adding up the weighted contributions of all trees
        for tree in self.trees:
            final_predictions += self.learning_rate * tree.predict(X)

        # return raw regression values
        return final_predictions

    # aggregating feature usage counts from all trees and print the top predictors
    def get_feature_importance(self, feature_names):
        total_counts = {}

        # looping every tree in the model
        for tree in self.trees:
            # access the counts we stored in the Tree object during fit()
            for index, count in tree.feature_counts.items():
                if index in total_counts:
                    total_counts[index] += count
                else:
                    total_counts[index] = count

        # sorting by importance
        sorted_features = sorted(total_counts.items(), key=lambda x: x[1], reverse=True)

        # results
        print("\nFeature Importance")
        for i, (feat_idx, count) in enumerate(sorted_features[:10]):
            name = feature_names[feat_idx]  # Convert Index to Name
            print(f"{i + 1}. {name}: Used to split {count} times")

if __name__ == "__main__":
    # loading data
    print("Loading clean data...")
    df = pd.read_csv("clean_data.csv")

    X = df.drop("TARGET", axis=1).values
    y = df["TARGET"].values

    # train test split
    indices = np.random.permutation(len(X))
    test_size = int(len(X) * 0.2)
    test_idx, train_idx = indices[:test_size], indices[test_size:]
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    print("Training model...")
    gbm = CustomGradientBoosting(n_estimators=10, learning_rate=0.1, max_depth=3)
    gbm.fit(X_train, y_train)

    print("Evaluating...")
    probs = gbm.predict(X_test)

    # threshold 0.5
    preds = (probs > 0.5).astype(int)
    acc = np.mean(preds == y_test)
    print(f"\nFinal test accuracy: {acc * 100:.2f}%")

    auc = roc_auc_score(y_test, probs)
    print(f"Final AUC score: {auc:.4f}")

    # feature importance
    feature_names = df.drop("TARGET", axis=1).columns
    gbm.get_feature_importance(feature_names)

    # sklearn model
    sk_model = GradientBoostingClassifier(n_estimators=10, learning_rate=0.1, max_depth=3)
    sk_model.fit(X_train, y_train)
    sk_probs = sk_model.predict_proba(X_test)[:, 1]
    sk_auc = roc_auc_score(y_test, sk_probs)

    print(f"Model's AUC: {auc:.4f}")
    print(f"Sklearn AUC:   {sk_auc:.4f}")

    # get class predictions (0 or 1)
    y_pred_class = (probs > 0.5).astype(int)

    # generate matrix
    cm = confusion_matrix(y_test, y_pred_class)

    # plot
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.savefig("confusion_matrix.png")
    plt.show()

    best_thresh = 0
    best_f1 = 0

    # trying thresholds from 0.05 to 0.5
    for thresh in np.arange(0.05, 0.50, 0.01):
        preds = (probs > thresh).astype(int)
        f1 = f1_score(y_test, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh

    print(f"Optimal threshold: {best_thresh:.2f}")

    # plot matrix with new threshold
    y_pred_class = (probs > best_thresh).astype(int)
    cm = confusion_matrix(y_test, y_pred_class)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel(f'Predicted Label (Threshold > {best_thresh:.2f})')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix - Optimized Threshold')
    plt.savefig("confusion_matrix_optimized.png")
    plt.show()

    # calculating roc coordinates
    fpr, tpr, thresholds = roc_curve(y_test, probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guessing')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)

    plt.savefig("roc_curve.png")
    plt.show()