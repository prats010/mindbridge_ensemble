"""Train base models + 4 ensemble techniques (Bagging, Boosting, Voting, Stacking)
for distress prediction and risk-level classification."""
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (BaggingClassifier, RandomForestClassifier,
                               AdaBoostClassifier, GradientBoostingClassifier,
                               VotingClassifier, StackingClassifier)
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

FEATURES = ["sleep_hours", "mood_score", "stress_level", "anxiety_score",
            "social_activity", "screen_time", "negative_thoughts"]


def load_data(path="mindbridge.csv"):
    df = pd.read_csv(path)
    X = StandardScaler().fit_transform(df[FEATURES])
    return df, X


def get_models():
    base = DecisionTreeClassifier(max_depth=4, random_state=42)
    return {
        "Logistic Regression": LogisticRegression(),
        "Decision Tree": DecisionTreeClassifier(max_depth=4, random_state=42),
        "Bagging (Random Forest)": RandomForestClassifier(n_estimators=100, random_state=42),
        "Bagging (manual)": BaggingClassifier(estimator=base, n_estimators=50, random_state=42),
        "Boosting (AdaBoost)": AdaBoostClassifier(estimator=base, n_estimators=50, random_state=42),
        "Boosting (Gradient Boosting)": GradientBoostingClassifier(random_state=42),
        "Voting (Hard)": VotingClassifier(estimators=[
            ("lr", LogisticRegression()),
            ("dt", DecisionTreeClassifier(max_depth=4, random_state=42)),
            ("rf", RandomForestClassifier(n_estimators=50, random_state=42)),
        ], voting="hard"),
        "Stacking": StackingClassifier(estimators=[
            ("lr", LogisticRegression()),
            ("dt", DecisionTreeClassifier(max_depth=4, random_state=42)),
            ("rf", RandomForestClassifier(n_estimators=50, random_state=42)),
        ], final_estimator=LogisticRegression()),
    }


def run(target_col, X, df, label):
    y = df[target_col]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"\n========== Target: {label} ==========")
    for name, model in get_models().items():
        model.fit(Xtr, ytr)
        pred = model.predict(Xte)
        acc = accuracy_score(yte, pred)
        print(f"\n--- {name} ---")
        print("Accuracy:", round(acc, 3))
        print("Confusion Matrix:\n", confusion_matrix(yte, pred))


if __name__ == "__main__":
    df, X = load_data()
    run("distress", X, df, "Distress (binary)")
    run("risk_level", X, df, "Risk Level (Low/Medium/High)")
