# MindBridge AI — Context-Aware Mental Health Support (Ensemble Learning)

Predicts user mental-health distress and risk level using base ML
models and four ensemble techniques: **Bagging, Boosting, Voting,
Stacking**.

## 1. Problem
Mental-health distress changes dynamically over time. Generic chatbots
lack persistent emotional context. Crisis signals need rapid
identification and escalation.

## 2. What It Predicts
| Target | Type | Values |
|---|---|---|
| `distress` | Binary classification | 0 (No) / 1 (Yes) |
| `risk_level` | Multi-class classification | Low / Medium / High |

## 3. Dataset
Synthetic survey-style dataset (300 users), generated in
`generate_data.py`, saved as `mindbridge.csv`.

**Features:** sleep_hours, mood_score, stress_level, anxiety_score,
social_activity, screen_time, negative_thoughts

## 4. Pipeline
```
Raw features → StandardScaler → Train/Test Split (80/20)
            → Base + Ensemble Models → Accuracy / Confusion Matrix
            → Best model selected
```

## 5. Models

**Base models**
- Logistic Regression
- Decision Tree

**Ensemble models**
| Technique | Implementation | Idea |
|---|---|---|
| Bagging | `RandomForestClassifier`, `BaggingClassifier` | Many trees on random data subsets, averaged |
| Boosting | `AdaBoostClassifier`, `GradientBoostingClassifier` | Models built sequentially, each fixing prior errors |
| Voting | `VotingClassifier` (hard) | LR + DT + RF vote, majority wins |
| Stacking | `StackingClassifier` | LR + DT + RF outputs fed into a final Logistic Regression |

## 6. Sample Results
(synthetic data, your numbers will vary slightly)

| Target | Best model | Accuracy |
|---|---|---|
| Distress | Logistic Regression | 1.00 |
| Risk Level | Stacking | 0.93 |

Full comparison saved to `results.csv` after running `evaluate.py`.

## 7. Project Structure
```
├── generate_data.py   # builds synthetic dataset
├── train_models.py    # base + ensemble models, per-model report
├── evaluate.py         # runs all models, saves results.csv, picks best
├── requirements.txt
└── README.md
```

## 8. Setup & Run
```bash
pip install -r requirements.txt
python generate_data.py     # creates mindbridge.csv
python train_models.py      # prints accuracy + confusion matrix per model
python evaluate.py          # saves results.csv + best model per target
```

## 9. Why Ensembles
- Bagging reduces variance (overfitting) by averaging many models.
- Boosting reduces bias by correcting previous models' mistakes.
- Voting combines diverse models for a more stable prediction.
- Stacking learns the best way to combine model outputs.

## 10. Limitations
- Dataset is synthetic, not real clinical data.
- Small sample size (300 rows) → results won't generalize.

## 11. Future Scope
- Use real, larger mental-health datasets.
- Add time-series mood tracking (dynamic distress).
- Build a web/mobile dashboard for users.
- Add crisis escalation to counselors / helplines.
- Add NLP-based text analysis / chatbot support.

## Author
Prathamesh Anil Bhamare
