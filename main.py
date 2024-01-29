import sklearn
from sklearn import datasets
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, BaggingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from pprint import pprint

# Function to evaluate model accuracy
def evaluate_model_accuracy(x_test_data, y_test_data, model, cv_splits):
    kfold_splits = KFold(n_splits=cv_splits)
    accuracies = []
    for _, test_index in kfold_splits.split(x_test_data):
        x_test_fold = x_test_data[test_index]
        y_test_fold = y_test_data[test_index]
        accuracies.append(accuracy_score(y_test_fold, model.predict(x_test_fold)))
    return accuracies

# K-Nearest Neighbors Classifier
def k_nearest_neighbors(train_x, train_y, test_x, test_y):
    neighbor_values = np.arange(1, 21)
    grid_params = {'n_neighbors': neighbor_values}
    knc = KNeighborsClassifier()
    grid_search = GridSearchCV(knc, grid_params, cv=10)
    grid_search.fit(train_x, train_y)
    evaluate_model_accuracy(test_x, test_y, grid_search, 10)
    print(f"Optimal K value: {grid_search.best_params_['n_neighbors']} with accuracy: {grid_search.best_score_}")

# Multi-layer Perceptron Classifier
def mlp_classifier(train_x, train_y, test_x, test_y):
    mlp = MLPClassifier(hidden_layer_sizes=(100, 100), activation='relu', solver='sgd', learning_rate='constant', learning_rate_init=0.02, max_iter=500, tol=1e-4)
    mlp.fit(train_x, train_y)
    mlp_accuracy = mlp.score(test_x, test_y)
    print(f"MLP Classifier -> Accuracy: {mlp_accuracy}")

# AdaBoost Classifier
def ada_boost_classifier(train_x, train_y, test_x, test_y):
    ada_boost = AdaBoostClassifier(n_estimators=100)
    ada_boost.fit(train_x, train_y)
    ada_boost_scores = cross_val_score(ada_boost, test_x, test_y, cv=10)
    print(f"AdaBoost -> Average accuracy: {ada_boost_scores.mean()}")

# Bagging Classifier
def bagging_classifier(train_x, train_y, test_x, test_y):
    bagging = BaggingClassifier(n_estimators=100)
    bagging.fit(train_x, train_y)
    bagging_scores = cross_val_score(bagging, test_x, test_y, cv=10)
    print(f"Bagging -> Average accuracy: {bagging_scores.mean()}")

# Gradient Boosting Classifier
def gradient_boosting_classifier(train_x, train_y, test_x, test_y):
    gradient_boost = GradientBoostingClassifier(n_estimators=100, learning_rate=0.6, max_depth=1, random_state=0)
    gradient_boost.fit(train_x, train_y)
    gradient_boost_accuracy = gradient_boost.score(test_x, test_y)
    print(f"Gradient Boosting -> Accuracy: {gradient_boost_accuracy}")

if __name__ == '__main__':
    # Load and preprocess the digits dataset
    digits_dataset = datasets.load_digits()
    X_digits, Y_digits = digits_dataset.data, digits_dataset.target
    X_train_digits, X_test_digits, Y_train_digits, Y_test_digits = train_test_split(X_digits, Y_digits, test_size=0.3)
    X_train_digits /= np.max(X_train_digits)
    X_test_digits /= np.max(X_test_digits)

    # Apply classifiers to the digits dataset
    k_nearest_neighbors(X_train_digits, Y_train_digits, X_test_digits, Y_test_digits)
    mlp_classifier(X_train_digits, Y_train_digits, X_test_digits, Y_test_digits)
    ada_boost_classifier(X_train_digits, Y_train_digits, X_test_digits, Y_test_digits)
    bagging_classifier(X_train_digits, Y_train_digits, X_test_digits, Y_test_digits)
    gradient_boosting_classifier(X_train_digits, Y_train_digits, X_test_digits, Y_test_digits)

    # Work with a different dataset: 20 newsgroups
    newsgroups_data_train = datasets.fetch_20newsgroups(subset='train')
    newsgroups_data_test = datasets.fetch_20newsgroups(subset='test')
    pprint(newsgroups_data_train.target_names)
    pprint(newsgroups_data_test.target_names)

    tfidf_vectorizer = TfidfVectorizer()
    train_vectors = tfidf_vectorizer.fit_transform(newsgroups_data_train.data)
    knc_newsgroups = KNeighborsClassifier()
    knc_newsgroups.fit(train_vectors, newsgroups_data_train.target)
    test_vectors = tfidf_vectorizer.transform(newsgroups_data_test.data)
    print(classification_report(newsgroups_data_test.target, knc_newsgroups.predict(test_vectors)))