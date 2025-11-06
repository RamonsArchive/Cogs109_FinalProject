# Overdispersion & Negative Binomial Regression

## Your Question: Do I Need Negative Binomial Instead of Poisson?

### Quick Answer: **YES!** ✅

Your data shows **overdispersion**: Variance (4.16) > Mean (2.99), with variance-to-mean ratio = **1.39**

---

## Understanding the Concepts

### 1. Variance vs Range

**These are DIFFERENT things:**

- **Range** = Max - Min = 10 - 0 = **10**

  - Just the spread of observed values
  - Doesn't tell us about overdispersion

- **Variance** = σ² = **4.16**

  - Average squared deviation from mean
  - Measures how spread out the data is around the mean
  - Formula: `Var(X) = E[(X - μ)²]`

- **Standard Deviation** = σ = **2.04**
  - Square root of variance
  - In same units as the data

### 2. What is Overdispersion?

**Poisson Assumption:**

- Variance = Mean (called "equidispersion")
- For Poisson distribution: E[X] = Var(X) = λ

**Your Data:**

- Mean = 2.99 injuries/game
- Variance = 4.16 injuries²/game
- **Variance > Mean** → OVERDISPERSION!

**What this means:**

- Injury counts are MORE variable than Poisson expects
- Poisson regression will **underestimate uncertainty**
- Standard errors will be too small → p-values too optimistic
- Negative Binomial is more appropriate

---

## Model Comparison

### Standard Poisson Regression

**Assumptions:**

- Mean = Variance (equidispersion)
- Only ONE parameter: λ (rate)

**Distribution:**

```
P(Y = k) = (λ^k * e^(-λ)) / k!
```

**When to use:**

- Count data
- Variance ≈ Mean
- Events are independent

**Your case:** ❌ Violated (Variance > Mean)

---

### Negative Binomial Regression ⭐

**Assumptions:**

- Allows Variance > Mean (overdispersion)
- TWO parameters: μ (mean) and α (dispersion)

**Variance formula:**

```
Var(Y) = μ + α*μ²
```

**Key insight:**

- When α = 0 → Negative Binomial = Poisson
- When α > 0 → Allows extra variance (overdispersion)

**Your case:** ✅ Appropriate!

---

### Quasi-Poisson Regression

**Similar to Negative Binomial but:**

- Doesn't assume a specific distribution
- Uses "quasi-likelihood" instead of full likelihood
- Also has dispersion parameter φ

**Variance formula:**

```
Var(Y) = φ * μ
```

**Differences from Negative Binomial:**

- Negative Binomial: Full likelihood model (can use AIC/BIC)
- Quasi-Poisson: Pseudo-model (cannot use AIC/BIC for comparison)

**Recommendation:** Use **Negative Binomial** because:

- ✅ Can compare models with AIC/BIC
- ✅ More interpretable
- ✅ Better for prediction intervals

---

## Comparison Table

| Feature                    | Linear Regression               | Poisson Regression   | Negative Binomial        | Quasi-Poisson     |
| -------------------------- | ------------------------------- | -------------------- | ------------------------ | ----------------- |
| **Data Type**              | Continuous                      | Count (0, 1, 2, ...) | Count                    | Count             |
| **Assumptions**            | Normal errors                   | Variance = Mean      | Variance > Mean          | Variance > Mean   |
| **Parameters**             | β (coefficients)                | λ (rate)             | μ (mean), α (dispersion) | μ, φ (dispersion) |
| **Handles Overdispersion** | N/A                             | ❌ No                | ✅ Yes                   | ✅ Yes            |
| **Can use AIC/BIC**        | ✅ Yes                          | ✅ Yes               | ✅ Yes                   | ❌ No             |
| **Prediction Range**       | (-∞, +∞)                        | [0, +∞)              | [0, +∞)                  | [0, +∞)           |
| **Your Data Fit**          | ⚠️ Poor (counts not continuous) | ❌ Overdispersed     | ✅ Best fit              | ✅ Good fit       |

---

## Your Data Analysis

### Distribution:

```
Injuries | Count | Percentage
---------|-------|------------
0        | 117   | 10.6%
1        | 161   | 14.6%
2        | 207   | 18.7%
3        | 228   | 20.6%  ← Most common
4        | 156   | 14.1%
5        | 104   | 9.4%
6        | 67    | 6.1%
7        | 33    | 3.0%
8        | 19    | 1.7%
9        | 10    | 0.9%
10       | 3     | 0.3%
```

### Key Statistics:

- **Mean**: 2.99 injuries/game
- **Variance**: 4.16
- **Overdispersion ratio**: 1.39 (39% more variance than Poisson expects)

### Why Overdispersion Exists:

1. **Unobserved heterogeneity**: Some games are inherently riskier

   - Playoff games vs regular season
   - Weather conditions
   - Team playing styles

2. **Clustering**: Injuries might occur in bursts

   - One dangerous play can injure multiple players
   - Certain game situations (e.g., goal-line stands) are riskier

3. **Omitted variables**: Factors we don't measure
   - Referee leniency
   - Player fatigue levels
   - Field conditions

---

## Why Predicted Range is Narrow (2-4)

You asked: "I'm only predicting 2 to 4 at max, why?"

### This is EXPECTED and GOOD! Here's why:

1. **Regression = Mean Prediction**

   - Models predict the **expected value** (average)
   - Not trying to predict every extreme case
   - Mean is 2.99, so predictions cluster around 2-4

2. **Extreme values are rare**

   - 10 injuries only happened 3 times (0.3%)
   - 0 injuries only happened 117 times (10.6%)
   - Most games (63%) have 1-4 injuries

3. **Model uncertainty**

   - Narrow range means model has HIGH uncertainty
   - Low R² (27% at best) confirms this
   - Injuries are largely unpredictable from game-level features

4. **This is better than predicting extremes!**
   - If model predicted 0-10 range, it would be overfitting
   - Predicting the mean (2-4) is statistically optimal when uncertain

### Prediction Interval vs Point Estimate:

- **Point estimate**: 2-4 range (what your model predicts)
- **Prediction interval**: Would be much wider (e.g., 0-8 with 95% confidence)
- Your model gives point estimates, which should be close to the mean

---

## Implementation Plan

### Adding Negative Binomial to Your Pipeline

I'll add `NegativeBinomial` as a 4th model type alongside:

1. **Poisson** - Count data (equidispersion)
2. **Linear** - Log-transformed target
3. **Logistic** - Binary classification
4. **Negative Binomial** ⬅️ NEW! - Count data with overdispersion

### Expected Improvements:

✅ **Better standard errors** - More realistic uncertainty estimates
✅ **Better fit statistics** - Lower deviance
✅ **More honest p-values** - Won't claim significance when there isn't
✅ **Similar predictions** - Point estimates will be similar to Poisson, but with correct uncertainty

---

## Summary

### Key Takeaways:

1. ✅ **You HAVE overdispersion** (Variance = 4.16 > Mean = 2.99)
2. ✅ **Negative Binomial is better than Poisson** for your data
3. ✅ **Narrow prediction range (2-4) is EXPECTED** - it's the mean!
4. ✅ **Variance ≠ Range** - Variance measures spread around mean, not min-max

### What I'll Implement:

1. Add `NegativeBinomialRegressor` to `make_pipeline()`
2. Train Negative Binomial models (Models 1-7)
3. Compare Negative Binomial vs Poisson performance
4. Reorganize plot saving structure (best/non_best/exploratory)

Your narrow prediction range is **not a problem** - it reflects the reality that injuries are hard to predict from game-level features! The Negative Binomial model will give you more honest uncertainty estimates. 🎯
