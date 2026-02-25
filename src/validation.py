from sklearn.model_selection import cross_val_score
import numpy as np

def cross_validate_model(model, X, y, cv=5):
    """
    Performs K-Fold cross validation.
    """
    scores = cross_val_score(
        model,
        X,
        y,
        cv=cv,
        scoring="r2"
    )

    return {
        "CV Mean R2": np.mean(scores),
        "CV Std R2": np.std(scores)
    }