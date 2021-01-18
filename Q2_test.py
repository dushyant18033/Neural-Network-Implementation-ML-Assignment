from Q1 import MyNeuralNetwork
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split


# importing dataset from openml
mnist=fetch_openml('mnist_784')
X=mnist.data
y=mnist.target
X=X.astype(dtype='float32')
y=y.astype(dtype='int32')

# 80-20 train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)


model = MyNeuralNetwork(5, [784, 256, 128, 64, 10], activation='relu', learning_rate=0.01, weight_init='normal', batch_size=7000, num_epochs=100)
model = model.fit(X_train,y_train, Xtest=X_test, ytest=y_test, save_error=True)

print("Train Acc:",model.score(X_train,y_train))
print("Test Acc:",model.score(X_test,y_test))