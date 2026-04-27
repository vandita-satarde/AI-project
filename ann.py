from sklearn.neural_network import MLPClassifier

def train_ann(X, y):
    model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500)
    model.fit(X, y)
    return model