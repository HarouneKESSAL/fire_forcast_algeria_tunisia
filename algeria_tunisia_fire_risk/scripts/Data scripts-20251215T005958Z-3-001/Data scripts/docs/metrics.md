# Metrics Reference

Brief explanations of key classification metrics used in our imbalanced-learning evaluations: how each is calculated and how to interpret it.

## Confusion Matrix
- **What:** Counts of predictions vs. true labels: True Positive (TP), False Positive (FP), True Negative (TN), False Negative (FN).
- **Why:** Foundation for many metrics; helps understand error types.
- **Interpretation:**
  - FP high → many false alarms; TN low → poor background filtering.
  - FN high → missed events; TP low → weak detection.

## Precision
- **Formula:** `Precision = TP / (TP + FP)`
- **What:** Of all predicted positives, how many are truly positive.
- **Interpretation:** High precision means few false alarms; can drop if the model is aggressive at flagging positives.
- **Note (imbalance):** Precision is sensitive to FP; useful when false alarms are costly.

## Recall (Sensitivity)
- **Formula:** `Recall = TP / (TP + FN)`
- **What:** Of all actual positives, how many the model correctly finds.
- **Interpretation:** High recall means few misses; can drop if the model is conservative.
- **Note (imbalance):** Recall is critical when missing a positive is costly.

## F1 Score
- **Formula:** `F1 = 2 * (Precision * Recall) / (Precision + Recall)`
- **What:** Harmonic mean of precision and recall.
- **Interpretation:** Balances false alarms and misses; useful single summary when both matter.
- **Note:** Harmonic mean penalizes extreme imbalance between precision and recall.

## ROC-AUC
- **ROC Curve:** Plots True Positive Rate (Recall) vs. False Positive Rate (`FPR = FP / (FP + TN)`) as the decision threshold varies.
- **AUC:** Area under the ROC curve (between 0 and 1).
- **Interpretation:** Probability the classifier ranks a random positive above a random negative; threshold-independent.
- **Note (imbalance):** ROC-AUC can look good even with many false positives when negatives dominate; complement with PR-AUC.

## PR-AUC (Average Precision)
- **PR Curve:** Plots Precision vs. Recall across thresholds.
- **AP / PR-AUC:** Area under the PR curve (often computed as Average Precision).
- **Interpretation:** Captures performance on the positive class under imbalance; higher AP indicates better precision-recall tradeoff.
- **Note:** More informative than ROC-AUC for heavily imbalanced data.

## Threshold Tuning
- **What:** Choose a probability threshold (instead of default 0.5) to meet objectives (e.g., maximize F1, or achieve a target precision/recall).
- **How:** Use the PR curve to select thresholds that optimize F1 or satisfy constraints (e.g., recall ≥ X% with highest precision).
- **Interpretation:** Adjusts trade-off between false alarms and misses according to domain priorities.

## Practical Guidance
- Report both **ROC-AUC** and **PR-AUC** for imbalanced datasets.
- Include **Precision, Recall, F1** and the **confusion matrix** to understand error types.
- Use **StratifiedKFold** to compute mean ± std across folds for robust estimates.
- Align threshold to the operational cost of FP vs. FN (e.g., prioritize recall if missing fires is costlier, or precision if false alarms are costly).
