from sklearn.model_selection import train_test_split
from knn import KNN
import pandas as pd
from csv import reader
import matplotlib as plt

def load_csv(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset

def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())
 
def str_column_to_int(dataset, column):
    class_values = [row[column] for row in dataset]
    unique = set(class_values)
    lookup = dict()
    for i, value in enumerate(unique):
        lookup[value] = i
    
    for row in dataset:
        row[column] = lookup[row[column]]
    return lookup

def main():
    df = pd.read_csv('./data/iris.csv', header=None)
    # dataset = load_csv('./data/iris.csv')
    df.columns = [ 'SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm', 'Species']
    X = df.iloc[:, :3]
    y = df['Species']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    accuracies = []
    ks = range(1, 6)
    for k in ks:
        knn = KNN(k=k)
        knn.fit(X_train, y_train)
        accuracy = knn.evaluate(X_test, y_test)
        accuracies.append(accuracy)
    
    fig, ax = plt.subplots()
    ax.plot(ks, accuracies)
    ax.set(xlabel="k", ylabel="Accuracy", title="Performance of knn")
    plt.show()

    """ for i in range(len(dataset[0])-1):
        str_column_to_float(dataset, i)

    str_column_to_int(dataset, len(dataset[0])-1)
    num_neighbors = [1, 3, 5]

    row = [5.7,7.9,9.2,1.3]
    label = knn.prediction_classification(dataset, row, num_neighbors)
    print('Data=%s, Predicted: %s' % (row, label)) """

if __name__ == '__main__':
    main()