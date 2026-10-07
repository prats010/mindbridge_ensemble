"""Run every model on both targets, save a comparison table, print the best model."""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from train_models import load_data, get_models

TARGETS = {"distress": "Distress (binary)", "risk_level": "Risk Level (L/M/H)"}


def main():
    df, X = load_data()
    rows = []
    for col, label in TARGETS.items():
        y = df[col]
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
        for name, model in get_models().items():
            model.fit(Xtr, ytr)
            acc = accuracy_score(yte, model.predict(Xte))
            rows.append({"target": label, "model": name, "accuracy": round(acc, 3)})

    results = pd.DataFrame(rows)
    results.to_csv("results.csv", index=False)
    print(results.to_string(index=False))

    print("\nBest model per target:")
    print(results.loc[results.groupby("target")["accuracy"].idxmax()].to_string(index=False))


if __name__ == "__main__":
    main()
