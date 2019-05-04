def calc_loss(y_true, y_pred):
    return np.mean((y_pred - y_true) ** 2)

def calc_loss_non_vectorized(y_true, y_pred):
    total = 0
    for tru, pred in zip(y_true, y_pred):
        total += (pred[0] - tru[0]) ** 2
    return total/len(y_pred)