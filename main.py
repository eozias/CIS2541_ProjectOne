# Import Statements
from sklearn import datasets
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import neighbors, metrics
from sklearn import naive_bayes
import seaborn as sns

def projectOne():
    # step 1 of the assignment: load the dataframe
    iris = datasets.load_iris()
    dataframe = pd.DataFrame(data = iris.data, columns=iris.feature_names)
    dataframe['target'] = iris.target
    print(dataframe.head())
    print(dataframe.tail())

    # step 2 of the assignment: visualize the data
    sns.pairplot(dataframe, height = 1.5, hue='target')
    plt.show()

    # step 3 of the assignment: split the data into training and testing sets
    iris_train_ftrs, iris_test_ftrs, iris_train_tgt, iris_test_tgt = train_test_split(iris.data, iris.target, test_size=0.25, random_state=42)
    print("Train features shape: ", iris_train_ftrs.shape)
    print("Test features shape: ", iris_test_ftrs.shape)

    # step 4 of the assignment: build and evaluate the kNN model
    knn = neighbors.KNeighborsClassifier(n_neighbors=3)
    knn.fit(iris_train_ftrs, iris_train_tgt)
    knn_pred = knn.predict(iris_test_ftrs)
    knn_f1_score = metrics.f1_score(iris_test_tgt, knn_pred, average='micro')

    # step 5 of the assignment: build and evaluate the Naive Bayes model
    nb = naive_bayes.GaussianNB()
    nb.fit(iris_train_ftrs, iris_train_tgt)
    nb_pred = nb.predict(iris_test_ftrs)
    nb_f1_score = metrics.f1_score(iris_test_tgt, nb_pred, average='micro')

    # step 6 of the assignment: compare the model performance using the F1 score
    print("kNN F1 Score: ", knn_f1_score)
    print("Naive Bayes F1 Score: ", nb_f1_score)

if __name__ == "__main__":
    projectOne()