Let me walk through the entire anomaly detector script like you're learning it for the first time.

## The Big Picture

**What we're building:** A system that looks at login attempts and says "Does this look normal or weird?"

**How it works:** We show the computer thousands of normal logins, and it learns what "normal" looks like. Then when a new login happens, it checks if it fits the pattern or not.

---

## Part 1: Creating Fake Login Data

```python
np.random.seed(42)
```

**Why?** This makes our "random" data the same every time we run it. Good for testing!

```python
n_samples = 5000
```

**Why 5,000?** We need enough data for the computer to learn patterns, but not so much it takes forever to run.

---

### Creating Realistic Login Times

```python
hours = np.random.choice(range(24), n_samples, p=[
    0.02, 0.01, 0.01, 0.01, 0.01, 0.02,  # midnight-5am
    0.03, 0.04, 0.06, 0.08, 0.06, 0.05,  # 6am-11am
    ...
])
```

**What's happening?** We're creating 5,000 random hours, but weighted to be realistic:

- **2am gets 1%** of logins (people are sleeping)
- **9am gets 8%** of logins (people starting work)
- **8pm gets 8%** of logins (evening browsing)

**Why the weights?** If we just picked random hours (equal chance for each), we'd get the same number of logins at 3am as 9am. That's not realistic! Real people sleep at night and work during the day.

---

### Creating Countries

```python
countries = np.random.choice(
    ['US', 'UK', 'CA', 'DE', 'FR', 'AU'], 
    n_samples, 
    p=[0.4, 0.2, 0.15, 0.1, 0.1, 0.05]
)
```

**What's happening?** 40% from US, 20% from UK, etc.

**Why?** We're pretending this is a US-based app, so most users are American. If someone logs in from Russia or China (not in our list), that's unusual!

---

### Creating Devices

```python
devices = np.random.choice(
    ['mobile', 'desktop', 'tablet'], 
    n_samples, 
    p=[0.6, 0.35, 0.05]
)
```

**Why 60% mobile?** Most people use phones these days! Tablets are only 5% because they're becoming less popular.

---

### Login Success Rate

```python
login_success = np.random.choice([True, False], n_samples, p=[0.95, 0.05])
```

**Why 95% success?** Most people type their password correctly. If you see 10 failed logins in a row, something's fishy!

---

### Sessions Per Hour

```python
sessions = np.where(
    devices == 'mobile', 
    np.random.poisson(3, n_samples),
    ...
)
```

**What's `poisson`?** A fancy way to create realistic numbers that cluster around a value (like 3) but sometimes go higher or lower.

**Why different by device?**

- Mobile users: ~3 sessions (quick checks)
- Desktop users: ~5 sessions (longer work sessions)
- Tablet users: ~2 sessions (casual browsing)

---

### Adding Fake Attacks

```python
anomalies = pd.DataFrame({
    'hour_of_day': [3, 2, 22],
    'country': ['RU', 'CN', 'BR'],
    'sessions_per_hour': [15, 20, 25]  # Way too many!
})
```

**Why?** We're sneaking in 3 obvious attacks so we can test if the model catches them:

- Russia at 3am with 15 sessions
- China at 2am with 20 sessions  
- Brazil at 10pm with 25 sessions

These are NOT in our normal patterns (no RU/CN/BR countries, way too many sessions).

---

## Part 2: Encoding Data (Converting Text to Numbers)

```python
le_country = LabelEncoder()
df_encoded['country_encoded'] = le_country.fit_transform(df['country'])
```

**Why?** Computers can't understand "US" or "UK". They need numbers.

**How it works:**

- AU = 0
- CA = 1
- DE = 2
- FR = 3
- UK = 4
- US = 5

**Why we save the encoder?** Later when we check a new login from "UK", we need to convert it to 4 using the same mapping.

---

## Part 3: Train/Test Split

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, df['true_anomaly'], 
    test_size=0.2, 
    random_state=42
)
```

**Why split?** 

- **Training data (80%)**: Teach the model what normal looks like
- **Test data (20%)**: See if it learned correctly by testing on data it's never seen

**Analogy:** Like studying for a test (train) and then taking the actual exam (test). You can't use the same questions for both!

---

## Part 4: Isolation Forest - The Magic

```python
iso_forest = IsolationForest(
    contamination=0.01,
    random_state=42,
    n_estimators=100
)
```

**What is Isolation Forest?** Imagine a forest of decision trees. Each tree tries to separate data points by asking questions:

- "Is hour > 12?"
- "Is country = US?"
- "Is sessions > 5?"

**How it finds anomalies:** Normal data takes many questions to isolate (they're clustered together). Weird data gets isolated quickly (they're alone out in the wilderness).

**contamination=0.01**: We're telling it "expect about 1% of logins to be weird." So it flags the top 1% weirdest ones.

**n_estimators=100**: Build 100 different trees and average their opinions.

---

## Part 5: Training

```python
iso_forest.fit(X_train_df)
```

**What happens?** The model looks at 4,000 normal logins and learns:

- "Most people log in between 8am-8pm"
- "Most are from US/UK/CA"
- "Most use mobile"
- "Sessions are usually 2-5"

It doesn't memorize individual logins—it learns the **patterns**.

---

## Part 6: Testing

```python
test_predictions = iso_forest.predict(X_test_df)
test_scores = iso_forest.score_samples(X_test_df)
```

**What happens?** 

- The model looks at 1,001 logins it's NEVER seen before
- For each one, it says: "Normal (1) or Anomaly (-1)?"
- It also gives a score: more negative = more suspicious

---

## Part 7: Evaluation

```python
confusion_matrix(y_test, test_predictions_binary)
```

**What's a confusion matrix?** A scorecard showing:

- How many normal logins did you catch? (993)
- How many normal logins did you wrongly flag? (7 false alarms)
- How many attacks did you catch? (1)
- How many attacks did you miss? (0)

---

## Part 8: Feature Contributions - "Why is this suspicious?"

```python
def analyze_feature_contributions(row_data, model, feature_names, X_train_df):
    base_score = model.score_samples(row_df)[0]
    
    for i, feature in enumerate(feature_names):
        modified_df = row_df.copy()
        modified_df.iloc[0, i] = X_train_df[feature].median()
        new_score = model.score_samples(modified_df)[0]
        contributions[feature] = new_score - base_score
```

**What's happening?** For each suspicious login, we ask: "What if we changed one feature to be more normal?"

**Example:** 

- Original: UK tablet at midnight, failed login, 0 sessions → Score: -0.67 (suspicious)
- Change device to desktop: → Score: -0.61 (less suspicious)
- **Conclusion:** The tablet is making it suspicious!

We do this for each feature to see which one is the "smoking gun."

---

## Part 9: Checking a New Login

```python
new_login = pd.DataFrame({
    'hour_of_day': [3],
    'country': ['RU'],
    'sessions_per_hour': [15]
})
```

**What happens?**

1. Convert Russia to a number (RU → some encoded value)
2. Feed it to the model
3. Model says: "This is weird! Score: -0.73"
4. Feature analysis shows: "High sessions + Russia + 3am = 🚨"

---

## Why This Approach Works

**Normal logins** have common patterns:

- 9am from US on mobile with 3 sessions ✓
- 2pm from UK on desktop with 5 sessions ✓

**Weird logins** break the pattern:

- 3am from Russia with 15 sessions 🚨
- Midnight from China with 20 sessions 🚨

The model learns "normal" and flags anything that doesn't fit!

---

## Real-World Use

In production, you'd:

1. Train on millions of real user logins
2. Every new login gets scored in real-time
3. High scores (very negative) trigger alerts
4. Security team investigates suspicious logins
5. Update the model monthly with new data

<br>
