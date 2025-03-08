from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

class RandomForest:

    def __init__(self):
        self.model = None

    def fit(self, X_train, y_train):

        # Train Random Forest Model
        rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
        rf_model.fit(X_train, y_train)
        self.model = rf_model

    def predict(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        conf_matrix = confusion_matrix(y_test, y_pred)

        plt.figure(figsize=(6, 5))
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Negative", "Neutral", "Positive"], yticklabels=["Negative", "Neutral", "Positive"])
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title("Confusion Matrix for Random Forest Model")
        plt.savefig("RandomForest.png")
        plt.show()

        print("Confusion Matrix:")
        print(conf_matrix)
        print(classification_report(y_test, y_pred))
