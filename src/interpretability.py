from skope_rules import SkopeRules
import numpy as np
import pandas as pd

class RuleFitExplainer:
    def __init__(self, feature_names):
        """
        Wrapper for RuleFit (via SkopeRules).
        """
        self.feature_names = feature_names
        self.clf = SkopeRules(max_depth_duplication=None,
                              n_estimators=10,
                              precision_min=0.5,
                              recall_min=0.01,
                              feature_names=feature_names)

    def fit(self, X_train, y_pred_binary):
        """
        Train the surrogate model on original features X and 
        GNN predicted labels (or high-risk class).
        """
        self.clf.fit(X_train, y_pred_binary)

    def get_rules(self):
        return self.clf.rules_

    def predict(self, X):
        return self.clf.predict(X)
