import numpy as np

def sample(preds, temperature=1.0):
    preds=np.asarray(preds).astype("float64")
    preds=np.log(np.maximum(preds, 1e-10))/temperature
    exp_preds=np.exp(preds)
    preds=exp_preds / np.sum(exp_preds)
    probas=np.random.multinomial(1, preds, 1)
    return np.argmax(probas)