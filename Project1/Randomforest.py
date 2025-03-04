from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

class RandomForest:

    def __init__(self):
        self.model = None

    def fit(self, X_train, y_train):

        # Train Random Forest Model
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        self.model = rf_model

    def predict(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        print(classification_report(y_test, y_pred))
