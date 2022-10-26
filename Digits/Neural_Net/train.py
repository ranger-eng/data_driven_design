from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

digits = load_digits()

# pull training/testing data
res = train_test_split(digits.data, digits.target, train_size=.8, test_size=.2, random_state=1)
train_data, test_data, train_labels, test_labels = res

mlp = MLPClassifier(hidden_layer_sizes=(20,), max_iter=20000, activation='logistic', alpha=1e-5, solver='sgd', tol=1e-4, random_state=1, learning_rate_init=.3, verbose=True)

mlp.fit(train_data, train_labels)

predictions = mlp.predict(test_data)
a = accuracy_score(test_labels, predictions)
print(a)
