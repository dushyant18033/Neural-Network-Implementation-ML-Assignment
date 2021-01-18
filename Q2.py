from Q1 import MyNeuralNetwork
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import pickle
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from sklearn.neural_network import MLPClassifier


TRAIN_MODELS=False
PICKLE_TEST=False
LOSS_PLOT=False
TSNE_PLOT=True
SKLEARN_TEST=False


# importing dataset from openml
mnist=fetch_openml('mnist_784')
X=mnist.data
y=mnist.target
X=X.astype(dtype='float32')
y=y.astype(dtype='int32')

# 80-20 train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)


# Q2(1.1) for generating pickle model dumps
if TRAIN_MODELS:
    model = MyNeuralNetwork(5, [784, 256, 128, 64, 10], activation='sigmoid', learning_rate=0.1, weight_init='normal', batch_size=7000, num_epochs=100)
    model = model.fit(X_train,y_train, Xtest=X_test, ytest=y_test, save_error=True)
    pickle.dump(model, open("Weights/sigmoid","wb"))

    model = MyNeuralNetwork(5, [784, 256, 128, 64, 10], activation='relu', learning_rate=0.1, weight_init='normal', batch_size=7000, num_epochs=100)
    model = model.fit(X_train,y_train, Xtest=X_test, ytest=y_test, save_error=True)
    pickle.dump(model, open("Weights/relu","wb"))

    model = MyNeuralNetwork(5, [784, 256, 128, 64, 10], activation='tanh', learning_rate=0.1, weight_init='normal', batch_size=7000, num_epochs=100)
    model = model.fit(X_train,y_train, Xtest=X_test, ytest=y_test, save_error=True)
    pickle.dump(model, open("Weights/tanh","wb"))

    model = MyNeuralNetwork(5, [784, 256, 128, 64, 10], activation='linear', learning_rate=0.1, weight_init='normal', batch_size=7000, num_epochs=100)
    model = model.fit(X_train,y_train, Xtest=X_test, ytest=y_test, save_error=True)
    pickle.dump(model, open("Weights/linear","wb"))


# Q2(1.2) for test accuracy from pickle models
if PICKLE_TEST:
    model = pickle.load(open("Weights/tanh","rb"))
    print("TANH:")
    print("Train Acc:", model.score(X_train, y_train))
    print("Test Acc:", model.score(X_test, y_test))
    print()

    model = pickle.load(open("Weights/relu","rb"))
    print("RELU:")
    print("Train Acc:", model.score(X_train, y_train))
    print("Test Acc:", model.score(X_test, y_test))
    print()

    model = pickle.load(open("Weights/sigmoid","rb"))
    print("SIGMOID:")
    print("Train Acc:", model.score(X_train, y_train))
    print("Test Acc:", model.score(X_test, y_test))
    print()

    model = pickle.load(open("Weights/linear","rb"))
    print("LINEAR:")
    print("Train Acc:", model.score(X_train, y_train))
    print("Test Acc:", model.score(X_test, y_test))
    print()


# Q2(2) for loss vs epochs plot
if LOSS_PLOT:
    model = pickle.load(open("Weights/tanh","rb"))
    epochs = list(range(model.num_epochs))
    plt.clf()
    plt.plot(epochs, model.train_CE, label='train loss')
    plt.plot(epochs, model.test_CE, '--', label='test loss')
    plt.legend()
    plt.xlabel('Number of Iterations')
    plt.ylabel('Cross Entropy Loss')
    plt.suptitle("Cross Entropy Loss Vs Epochs | Tanh")
    plt.savefig("Plots/tanh.png")

    model = pickle.load(open("Weights/relu","rb"))
    epochs = list(range(model.num_epochs))
    plt.clf()
    plt.plot(epochs, model.train_CE, label='train loss')
    plt.plot(epochs, model.test_CE, '--', label='test loss')
    plt.legend()
    plt.xlabel('Number of Iterations')
    plt.ylabel('Cross Entropy Loss')
    plt.suptitle("Cross Entropy Loss Vs Epochs | ReLU")
    plt.savefig("Plots/relu.png")

    model = pickle.load(open("Weights/sigmoid","rb"))
    epochs = list(range(model.num_epochs))
    plt.clf()
    plt.plot(epochs, model.train_CE, label='train loss')
    plt.plot(epochs, model.test_CE, '--', label='test loss')
    plt.legend()
    plt.xlabel('Number of Iterations')
    plt.ylabel('Cross Entropy Loss')
    plt.suptitle("Cross Entropy Loss Vs Epochs | Sigmoid")
    plt.savefig("Plots/sigmoid.png")

    model = pickle.load(open("Weights/linear","rb"))
    epochs = list(range(model.num_epochs))
    plt.clf()
    plt.plot(epochs, model.train_CE, label='train loss')
    plt.plot(epochs, model.test_CE, '--', label='test loss')
    plt.legend()
    plt.xlabel('Number of Iterations')
    plt.ylabel('Cross Entropy Loss')
    plt.suptitle("Cross Entropy Loss Vs Epochs | Linear")
    plt.savefig("Plots/linear.png")


# Q2(5) t-SNE of final hidden layer features for best model
if TSNE_PLOT:
    model = pickle.load(open("Weights/tanh","rb"))
    
    last_layer = model.predict_proba(X_train, last_hidden=True)
    tsne = TSNE(n_components=2).fit_transform(last_layer)
    plt.figure(figsize=(20,10))
    for c in range(10):
        select = tsne[y_train==c]
        plt.scatter(select[:,0],select[:,1],label="Class "+str(c))
    plt.suptitle("t-SNE Analysis Train Data")
    plt.legend()
    plt.xlabel("Axis-1")
    plt.ylabel("Axis-2")
    plt.savefig("Plots/tsne_last_hidden_train.png")

    last_layer = model.predict_proba(X_test, last_hidden=True)
    tsne = TSNE(n_components=2).fit_transform(last_layer)
    plt.figure(figsize=(20,10))
    for c in range(10):
        select = tsne[y_test==c]
        plt.scatter(select[:,0],select[:,1],label="Class "+str(c))
    plt.suptitle("t-SNE Analysis Test Data")
    plt.legend()
    plt.xlabel("Axis-1")
    plt.ylabel("Axis-2")
    plt.savefig("Plots/tsne_last_hidden_test.png")


# Q2(6) sklearn accuracies
if SKLEARN_TEST:
    for acti in ['tanh', 'relu', 'logistic', 'identity']:
        model = MLPClassifier(
            hidden_layer_sizes = (256, 128, 64),
            activation = acti,
            solver = 'sgd',
            alpha = 0,
            batch_size = 7000,
            learning_rate = 'constant',
            learning_rate_init = 0.1,
            max_iter = 100,
            random_state=42
        )
        model.fit(X_train, y_train)
        print(acti)
        print("Train Acc:", model.score(X_train, y_train))
        print("Test Acc:", model.score(X_test, y_test))
        print()