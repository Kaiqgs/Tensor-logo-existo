from sklearn.datasets import load_boston  # Dataset;
from sklearn.linear_model import LinearRegression  # Sklearn for comparison;
from sklearn.preprocessing import StandardScaler  # Data preprocessing;
import numpy as np  # Linear algebra;

boston = load_boston()
# Features;
X = boston.data
# Labels;
y = boston.target.reshape(-1, 1)

# Scaling data;
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Instantiating parameters;
theta = np.random.randn(1, X.shape[1])
bias = np.zeros((1, y.shape[1]))


def forward_pass():
    return np.dot(X, theta.T) + bias


def calc_loss(y_true, y_pred):
    return np.mean((y_pred - y_true) ** 2)


def optimize(y_true, y_pred, lr=0.01):
    global theta, bias
    # Derivatives;
    dydl = -(np.sum(((y_true - y_pred) * 2), axis=1)/len(y))
    dtdl = X
    dbdl = 1
    # Gradients;
    tgrad = np.dot(dydl, dtdl)
    bgrad = np.dot(np.sum(dydl), dbdl)
    # Parameters update;
    theta -= tgrad.T * lr
    bias -= bgrad * lr


def train(epochs=1000):
    for i in range(epochs):
        f = forward_pass()
        optimize(y, f)
        if(i % 200 == 0):
            loss = calc_loss(y, f)
            print(f"Train: epoch[{i}] loss[{loss:.3f}]")
    loss = calc_loss(y, f)
    print(f"Homemade: performance[{loss:.3f}]")


def sklearn_compare():
    skmod = LinearRegression()
    skmod.fit(X, y)
    yhat = skmod.predict(X)
    loss = calc_loss(y, yhat)
    print(f"Scikit-Learn: performance[{loss:.3f}]")


# Homemade performance;
train()

# Sklearn performance;
sklearn_compare()
