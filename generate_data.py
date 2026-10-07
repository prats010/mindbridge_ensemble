"""Generate synthetic MindBridge AI dataset."""
import numpy as np, pandas as pd

np.random.seed(42)
n = 300
df = pd.DataFrame({
    "user_id": range(1, n + 1),
    "sleep_hours": np.random.normal(6, 1.5, n).clip(2, 10),
    "mood_score": np.random.randint(1, 11, n),
    "stress_level": np.random.randint(1, 11, n),
    "anxiety_score": np.random.randint(1, 11, n),
    "social_activity": np.random.randint(1, 11, n),
    "screen_time": np.random.normal(5, 2, n).clip(0, 14),
    "negative_thoughts": np.random.randint(0, 11, n),
})

risk_raw = (df.stress_level + df.anxiety_score + df.negative_thoughts
            - df.mood_score - df.social_activity)
df["distress"] = (risk_raw > risk_raw.median()).astype(int)
df["risk_level"] = pd.cut(risk_raw, bins=3, labels=["Low", "Medium", "High"])

df.to_csv("mindbridge.csv", index=False)
print("Saved mindbridge.csv with", n, "rows")
