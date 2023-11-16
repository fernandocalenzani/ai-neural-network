from sklearn.neural_network import MLPClassifier
from sklearn import datasets

iris_data = datasets.load_iris()
X_in = iris_data.data
Y_out = iris_data.target


nn = MLPClassifier(verbose=True, max_iter=100000, tol=0.00001, activation='logistic', learning_rate='adaptive', learning_rate_init=0.03, solver='adam')
nn.fit(X_in, Y_out)

flower = nn.predict([[5, 7.2, 5.1, 2.2]])

print(flower)
