import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================
# 1. GENERATE FAKE DATASET
# ============================================

# Linda: "Alright! We're setting the vibe so randomness is ✨repeatable✨"
# This makes sure every run gives the SAME random results (debug-friendly)
np.random.seed(42)

# Bob: "5,000 logins. Not too many. Not too few. Like a good lunch rush."
n_samples = 5000

# Tina: "These are hours of the day... but with probabilities... I like structure."
# np.random.choice():
# - Pick values from range(24) → hours 0–23
# - n_samples times
# - p = probability for EACH hour being chosen
# Translation: logins happen more during normal human hours, less at 3am
hours = np.random.choice(range(24), n_samples, p=[
    0.02, 0.01, 0.01, 0.01, 0.01, 0.02,  # 0-5am (people should be asleep)
    0.03, 0.04, 0.06, 0.08, 0.06, 0.05,  # 6-11am (coffee-powered humans)
    0.06, 0.05, 0.04, 0.04, 0.05, 0.06,  # 12-5pm (steady grind)
    0.08, 0.07, 0.06, 0.04, 0.03, 0.02   # 6-11pm (doomscroll hours)
])

# Gene: "Countries! Like people logging in from places! Geography rocks!"
# np.random.choice again, but for strings this time
countries = np.random.choice(
    ['US', 'UK', 'CA', 'DE', 'FR', 'AU'],
    n_samples,
    p=[0.4, 0.2, 0.15, 0.1, 0.1, 0.05]
)

# Louise: "Most people use phones. Tablets are for weirdos."
devices = np.random.choice(
    ['mobile', 'desktop', 'tablet'],
    n_samples,
    p=[0.6, 0.35, 0.05]
)

# Bob: "Most logins work. Because if they didn't, I'd hear about it."
login_success = np.random.choice([True, False], n_samples, p=[0.95, 0.05])

# Tina (deep breath): "Okay... this part is IMPORTANT."

# np.where(condition, value_if_true, value_if_false)
# Think: IF / ELSE but vectorized (runs on whole arrays at once)

# Translation:
# - If device is mobile → avg ~3 sessions/hour
# - If desktop → avg ~5 sessions/hour
# - Else (tablet) → avg ~2 sessions/hour
sessions = np.where(
    devices == 'mobile',
    np.random.poisson(3, n_samples),  # Poisson = "count-ish things"
    np.where(
        devices == 'desktop',
        np.random.poisson(5, n_samples),
        np.random.poisson(2, n_samples)
    )
)

# Linda: "Put it all in a nice little table!"
df = pd.DataFrame({
    'hour_of_day': hours,
    'country': countries,
    'device_type': devices,
    'login_success': login_success,
    'sessions_per_hour': sessions
})

# ============================================
# ADD ACTUAL SKETCHY STUFF
# ============================================

# Louise: "Now we add CRIME."
anomalies = pd.DataFrame({
    'hour_of_day': [3, 2, 22],
    'country': ['RU', 'CN', 'BR'],
    'device_type': ['desktop', 'desktop', 'mobile'],
    'login_success': [False, False, True],
    'sessions_per_hour': [15, 20, 25]  # Way too many sessions
})

df = pd.concat([df, anomalies], ignore_index=True)

# Bob: "Label the bad guys so we can see if the model figures it out."
# 0 = normal, 1 = anomaly
df['true_anomaly'] = 0
df.loc[df.index[-3:], 'true_anomaly'] = 1

print(f"Dataset created: {len(df)} rows, {df['true_anomaly'].sum()} true anomalies")

# ============================================
# 2. ENCODE CATEGORICAL FEATURES
# ============================================

# Tina: "Models don't speak words. They speak numbers."
df_encoded = df.copy()

# LabelEncoder:
# Turns categories into integers
# Example: ['US', 'UK', 'FR'] → [2, 1, 0] (order doesn't mean anything)
le_country = LabelEncoder()
le_device = LabelEncoder()
le_success = LabelEncoder()

# fit_transform():
# - fit = learn the mapping
# - transform = apply it
# Doing both at once
df_encoded['country_encoded'] = le_country.fit_transform(df['country'])
df_encoded['device_encoded'] = le_device.fit_transform(df['device_type'])
df_encoded['login_success_encoded'] = le_success.fit_transform(df['login_success'])

# These are the actual features the model sees
features = [
    'hour_of_day',
    'country_encoded',
    'device_encoded',
    'login_success_encoded',
    'sessions_per_hour'
]

X = df_encoded[features]

# ============================================
# 3. TRAIN / TEST SPLIT
# ============================================

# Bob: "Train on most of it. Test on some of it. Like tasting the burger."
X_train, X_test, y_train, y_test = train_test_split(
    X,
    df['true_anomaly'],
    test_size=0.2,
    random_state=42,
    stratify=df['true_anomaly']  # Keeps anomaly ratio consistent
)

# Keep indices so we can match predictions back to rows
train_indices = X_train.index
test_indices = X_test.index

# Keep DataFrames (IsolationForest is picky sometimes)
X_train_df = X.loc[train_indices]
X_test_df = X.loc[test_indices]

# ============================================
# 4. TRAIN ISOLATION FOREST
# ============================================

# Tina: "Isolation Forest works by isolating weird points quickly."
# Weird = fewer splits needed = suspicious
iso_forest = IsolationForest(
    contamination=0.005,  # Expect ~0.5% anomalies
    random_state=42,
    n_estimators=100      # Number of trees in the forest
)

# fit():
# Learn what "normal" looks like
iso_forest.fit(X_train_df)

# ============================================
# 5. PREDICT ON TEST SET
# ============================================

# predict():
# Returns:
#  1 = normal
# -1 = anomaly (yes, it's weird)
test_predictions = iso_forest.predict(X_test_df)

# score_samples():
# Lower score = more suspicious
test_scores = iso_forest.score_samples(X_test_df)

# Convert to sane labels
test_predictions_binary = (test_predictions == -1).astype(int)

# Save results back to original dataframe
df.loc[test_indices, 'predicted_anomaly'] = test_predictions_binary
df.loc[test_indices, 'anomaly_score'] = test_scores

# ============================================
# 6. EVALUATE
# ============================================

# Gene: "Numbers!!!"
print("\n" + "="*50)
print("CONFUSION MATRIX")
print("="*50)
cm = confusion_matrix(y_test, test_predictions_binary)
print(cm)

print("\n" + "="*50)
print("CLASSIFICATION REPORT")
print("="*50)
print(classification_report(y_test, test_predictions_binary))

# ============================================
# 7. ANALYZE FALSE POSITIVES
# ============================================

# Louise: "Who did we accuse that didn't deserve it?"
test_df = df.loc[test_indices].copy()

false_positives = test_df[
    (test_df['predicted_anomaly'] == 1) &
    (test_df['true_anomaly'] == 0)
]

true_positives = test_df[
    (test_df['predicted_anomaly'] == 1) &
    (test_df['true_anomaly'] == 1)
]

true_negatives = test_df[
    (test_df['predicted_anomaly'] == 0) &
    (test_df['true_anomaly'] == 0)
]

print("\n" + "="*50)
print(f"FALSE POSITIVES ({len(false_positives)} cases)")
print("="*50)
print(false_positives[['hour_of_day', 'country', 'device_type',
                       'login_success', 'sessions_per_hour', 'anomaly_score']])

print("\n" + "="*50)
print(f"TRUE POSITIVES ({len(true_positives)} cases)")
print("="*50)
print(true_positives[['hour_of_day', 'country', 'device_type',
                      'login_success', 'sessions_per_hour', 'anomaly_score']])

# ============================================
# 8. FEATURE CONTRIBUTION ANALYSIS
# ============================================

def analyze_feature_contributions(row_data, model, feature_names, X_train_df):
    """
    Bob voice: "What specifically made THIS login weird?"

    Strategy:
    - Score the row
    - Replace ONE feature with a normal (median) value
    - Re-score
    - If score improves a lot → that feature was suspicious
    """

    row_df = pd.DataFrame([row_data], columns=feature_names)
    base_score = model.score_samples(row_df)[0]

    contributions = {}

    for i, feature in enumerate(feature_names):
        # Create a copy with this feature set to median value
        modified_df = row_df.copy()
        modified_df.iloc[0, i] = X_train_df[feature].median()

        new_score = model.score_samples(modified_df)[0]
        # Positive contribution means removing it made it MORE normal
        contributions[feature] = new_score - base_score

    return contributions


# Analyze a few examples
print("\n" + "="*50)
print("FEATURE CONTRIBUTIONS TO ANOMALY DETECTION")
print("="*50)

if len(false_positives) > 0:
    print("\n--- FALSE POSITIVE EXAMPLE ---")
    fp_idx = false_positives.index[0]
    fp_data = X.loc[fp_idx].values
    fp_contributions = analyze_feature_contributions(fp_data, iso_forest, features, X_train_df)  # Added X_train_df

    print(f"Row: {df.loc[fp_idx][['hour_of_day', 'country', 'device_type', 'login_success', 'sessions_per_hour']].to_dict()}")
    print(f"Anomaly score: {df.loc[fp_idx, 'anomaly_score']:.4f}")
    print("\nFeature contributions (higher = more suspicious):")
    for feat, contrib in sorted(fp_contributions.items(), key=lambda x: x[1], reverse=True):
        print(f"  {feat}: {contrib:.4f}")

if len(true_positives) > 0:
    print("\n--- TRUE POSITIVE EXAMPLE ---")
    tp_idx = true_positives.index[0]
    tp_data = X.loc[tp_idx].values
    tp_contributions = analyze_feature_contributions(tp_data, iso_forest, features, X_train_df)  # Added X_train_df

    print(f"Row: {df.loc[tp_idx][['hour_of_day', 'country', 'device_type', 'login_success', 'sessions_per_hour']].to_dict()}")
    print(f"Anomaly score: {df.loc[tp_idx, 'anomaly_score']:.4f}")
    print("\nFeature contributions (higher = more suspicious):")
    for feat, contrib in sorted(tp_contributions.items(), key=lambda x: x[1], reverse=True):
        print(f"  {feat}: {contrib:.4f}")

# ============================================
# 9. VISUALS & NEW LOGIN TEST
# ============================================

# Final vibe:
# - Lower score = sketchy
# - Isolation Forest doesn't know "bad"
# - It only knows "weird"
# Plot 1: Anomaly Score Distribution
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.hist(test_df[test_df['true_anomaly'] == 0]['anomaly_score'],
         bins=30, alpha=0.7, label='Normal', color='green')
plt.hist(test_df[test_df['true_anomaly'] == 1]['anomaly_score'],
         bins=30, alpha=0.7, label='True Anomaly', color='red')
plt.xlabel('Anomaly Score')
plt.ylabel('Count')
plt.title('Anomaly Score Distribution')
plt.legend()
plt.axvline(x=-0.7, color='black', linestyle='--', label='Potential Threshold')

# Plot 2: False Positive Analysis - Hour of Day
plt.subplot(1, 3, 2)
if len(false_positives) > 0:
    fp_hours = false_positives['hour_of_day'].value_counts().sort_index()
    normal_hours = true_negatives['hour_of_day'].value_counts().sort_index()

    hours_range = range(24)
    fp_counts = [fp_hours.get(h, 0) for h in hours_range]
    normal_counts = [normal_hours.get(h, 0) for h in hours_range]

    x = np.arange(len(hours_range))
    width = 0.35
    plt.bar(x - width/2, normal_counts, width, label='Normal', alpha=0.7, color='green')
    plt.bar(x + width/2, fp_counts, width, label='False Positive', alpha=0.7, color='orange')
    plt.xlabel('Hour of Day')
    plt.ylabel('Count')
    plt.title('False Positives by Hour')
    plt.legend()
    plt.xticks(x[::2], hours_range[::2])

# Plot 3: False Positive Analysis - Country
plt.subplot(1, 3, 3)
if len(false_positives) > 0:
    fp_countries = false_positives['country'].value_counts()
    normal_countries = true_negatives['country'].value_counts()

    all_countries = sorted(set(fp_countries.index) | set(normal_countries.index))
    fp_counts = [fp_countries.get(c, 0) for c in all_countries]
    normal_counts = [normal_countries.get(c, 0) for c in all_countries]

    x = np.arange(len(all_countries))
    width = 0.35
    plt.bar(x - width/2, normal_counts, width, label='Normal', alpha=0.7, color='green')
    plt.bar(x + width/2, fp_counts, width, label='False Positive', alpha=0.7, color='orange')
    plt.xlabel('Country')
    plt.ylabel('Count')
    plt.title('False Positives by Country')
    plt.legend()
    plt.xticks(x, all_countries)

plt.tight_layout()
plt.savefig('anomaly_analysis.png', dpi=150, bbox_inches='tight')
print("\n✅ Visualization saved as 'anomaly_analysis.png'")

# ============================================
# 10. CHECK NEW LOGIN
# ============================================
print("\n" + "="*50)
print("TESTING NEW LOGIN")
print("="*50)

new_login = pd.DataFrame({
    'hour_of_day': [3],
    'country': ['RU'],
    'device_type': ['desktop'],
    'login_success': [False],
    'sessions_per_hour': [15]
})

new_login['country_encoded'] = le_country.transform(new_login['country'])
new_login['device_encoded'] = le_device.transform(new_login['device_type'])
new_login['login_success_encoded'] = le_success.transform(new_login['login_success'])

new_X = new_login[features]
prediction = iso_forest.predict(new_X)
score = iso_forest.score_samples(new_X)

print(f"New login: {new_login[['hour_of_day', 'country', 'device_type', 'login_success', 'sessions_per_hour']].iloc[0].to_dict()}")
print(f"Anomaly score: {score[0]:.4f}")

if prediction[0] == -1:
    print("🚨 ANOMALY DETECTED!")
else:
    print("✅ Normal login")

# Analyze what made it suspicious
new_contributions = analyze_feature_contributions(new_X.values[0], iso_forest, features, X_train_df)  # Added X_train_df
print("\nFeature contributions:")
for feat, contrib in sorted(new_contributions.items(), key=lambda x: x[1], reverse=True):
    print(f"  {feat}: {contrib:.4f}")

# if score < -0.75 and sessions_per_hour > 10:
#     alert = True
