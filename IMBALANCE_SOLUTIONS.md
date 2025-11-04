# Solutions for Class Imbalance in Logistic Regression

## Current Problem

- **Class Distribution**: 81% low-risk, 19% high-risk games
- **Current Threshold**: 0.5 (50% probability)
- **Result**: Model predicts almost everything as "low-risk"
- **Performance**: Recall = 2.4% (missing 97.6% of high-risk games!)

## Three Solutions (Already Implemented)

### ✅ Solution 1: Class Weights (ACTIVE)

**What it does**: Penalizes misclassifying high-risk games more heavily

**How it works**:

```python
# In model.py, line 47-51:
LogisticRegression(
    max_iter=5000,
    random_state=RANDOM_STATE,
    class_weight='balanced'  # ← This line!
)
```

**Effect**:

- High-risk errors cost ~4-5x more than low-risk errors
- Model tries harder to catch high-risk games
- Should increase recall (catch more dangerous games)
- May decrease precision (more false alarms)

**Trade-off**: More false positives, but better for safety!

---

### ✅ Solution 2: Lower Threshold (ACTIVE)

**What it does**: Classify as "high-risk" if probability > 30% (instead of 50%)

**How it works**:

```python
# In model.py, line 177-179 and 332-334:
classification_threshold = 0.3  # ← Adjust this!
y_pred = (y_pred_proba >= classification_threshold).astype(int)
```

**Current threshold**: 0.3 (30%)

**Effect of different thresholds**:
| Threshold | Recall ↑ | Precision ↓ | Use Case |
|-----------|----------|-------------|----------|
| 0.2 | Very High | Very Low | Maximum safety (catch almost all risks) |
| 0.3 | High | Low | Good balance for safety-critical |
| 0.4 | Medium | Medium | Moderate balance |
| 0.5 | Low | High | Default (poor for imbalanced data) |

**To adjust**: Change `classification_threshold = 0.3` to your preferred value

---

### Solution 3: SMOTE (Optional - Not Active)

**What it does**: Creates synthetic high-risk game examples

**How to use**:

```bash
# 1. Install library
pip install imbalanced-learn

# 2. Import in model.py
from balance import apply_smote

# 3. Add before training (in train_model function, after line 93):
if model_type == "logistic":
    X_train, y_train = apply_smote(X_train, y_train)
```

**Effect**:

- Creates balanced 50/50 dataset
- Model sees equal examples of both classes
- Can improve recall significantly

**Caution**: Can overfit to synthetic data

---

## What to Expect After Running

### Current Results (Threshold 0.5, No Class Weights):

- Accuracy: 80%
- Precision: 20%
- Recall: 2.4% ← **TERRIBLE**
- F1: 4.4%
- ROC-AUC: 57.6%

### Expected Results (Threshold 0.3 + Class Weights):

- Accuracy: ~70-75% ← Will decrease
- Precision: ~15-18% ← Will decrease slightly
- Recall: ~20-40% ← **Should improve 10-15x!**
- F1: ~18-25% ← Should improve significantly
- ROC-AUC: ~60-65% ← Might improve slightly

**Key Insight**:

- Accuracy will DROP (because we'll predict "high-risk" more often)
- Recall will RISE (we'll catch more actual high-risk games)
- **This is good!** For safety, catching dangerous games matters more than accuracy

---

## How to Test Different Configurations

### Test 1: Current Setup (Class Weights + Threshold 0.3)

```bash
cd "my-project/src"
python main.py
```

### Test 2: Try Threshold 0.2 (More aggressive)

1. Change line 177 in `model.py`: `classification_threshold = 0.2`
2. Change line 332 in `model.py`: `classification_threshold = 0.2`
3. Run: `python main.py`

### Test 3: Try Threshold 0.4 (Less aggressive)

1. Change both thresholds to `0.4`
2. Run: `python main.py`

### Test 4: Add SMOTE

1. Install: `pip install imbalanced-learn`
2. Uncomment SMOTE code (see Solution 3 above)
3. Run: `python main.py`

---

## Understanding ROC-AUC vs Threshold

**Important Distinction**:

1. **Classification Threshold** (0.3 in your code):

   - Used for **final predictions** (0 or 1)
   - Affects: Accuracy, Precision, Recall, F1
   - You control this manually

2. **ROC-AUC** (currently 57.6%):
   - Tests **all possible thresholds** (0.0 to 1.0)
   - Measures overall model quality
   - Independent of your chosen threshold
   - **0.5 = random guessing, 1.0 = perfect**

**Your ROC-AUC of 0.576 means**:

- Even with optimal threshold, model barely beats random
- Features don't strongly predict injury risk
- May need better features or accept prediction limits

---

## Recommendation

**For player safety**, use:

1. ✅ `class_weight='balanced'` (already active)
2. ✅ `classification_threshold = 0.3` (already active)
3. ⚠️ Monitor recall in results - aim for 30%+

**If recall is still too low**, try:

- Lower threshold to 0.2
- Add SMOTE
- Collect more high-risk game data
- Engineer new features (team injury history, player fatigue, etc.)

---

## Files Modified

- ✅ `model.py` - Added class weights and threshold adjustment
- ✅ `balance.py` - Created helper functions for SMOTE
- 📄 `IMBALANCE_SOLUTIONS.md` - This file
