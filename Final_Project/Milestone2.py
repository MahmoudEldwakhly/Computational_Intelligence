

# Project Milestone 2 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    confusion_matrix, classification_report, accuracy_score
)

import tensorflow as tf

# LLM
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

sns.set_context("notebook")
plt.rcParams["figure.figsize"] = (12, 5)

# ============================================================
# 0) SETTINGS
# ============================================================
FILE_PATH = "/content/Airbnb_Data.csv"   # Upload CSV to Colab or change path
RANDOM_SEED = 42

TRAIN_FRAC, VAL_FRAC, TEST_FRAC = 0.70, 0.15, 0.15

# Choose LLM model (large is better but heavier)
LLM_MODEL_NAME = "google/flan-t5-large"   # or "google/flan-t5-base" if GPU memory low
LLM_EVAL_SAMPLES = 200                    # LLM evaluation size (increase if you want)

# Optional: stabilize extreme outliers (recommended)
MIN_PRICE, MAX_PRICE = 20, 1500

# ============================================================
# 1) LOAD DATA
# ============================================================
df = pd.read_csv(FILE_PATH)
print(" Loaded:", df.shape)
display(df.head())

# ============================================================
# 2) CREATE TARGET PRICE
# ============================================================
if "log_price" in df.columns:
    df["Price"] = np.exp(df["log_price"])
elif "price" in df.columns:
    df["Price"] = df["price"]
else:
    raise ValueError("Target not found: need 'log_price' or 'price' column")

df = df[df["Price"].between(MIN_PRICE, MAX_PRICE)].copy()
df.reset_index(drop=True, inplace=True)
print(" After price filtering:", df.shape)

# ============================================================
# 3) WHY THIS IS REGRESSION (NOT CLASSIFICATION)
# ============================================================
unique_prices = df["Price"].nunique()
print(f"Unique Price values: {unique_prices} out of {len(df)} rows.")
print(
    "\n Explanation (use in report):\n"
    "- Price is continuous numeric with many unique values → main task is REGRESSION.\n"
    "- Classification is only possible after binning prices into categories (Low/Medium/High).\n"
)

# ============================================================
# 4) CATEGORY LABELS (FOR CONFUSION MATRIX + LLM CLASSIFICATION)
# ============================================================
def get_price_category(p):
    if p < 90: return "Low"
    elif p < 180: return "Medium"
    else: return "High"

df["Category"] = df["Price"].apply(get_price_category)

print("\nCategory balance (%):")
display(df["Category"].value_counts(normalize=True).mul(100).round(2).to_frame("percent"))

# ============================================================
# 5) PART A: DEEP EDA (LOTS OF TABLES + GRAPHS)
# ============================================================
print("\n==================== EDA ====================")

# Missing values table
miss_cnt = df.isna().sum().sort_values(ascending=False)
miss_pct = (miss_cnt / len(df) * 100).round(2)
miss_table = pd.DataFrame({"missing_count": miss_cnt, "missing_%": miss_pct})
display(miss_table.head(30))

# Price distribution
plt.figure(figsize=(12,4))
sns.histplot(df["Price"], bins=60, kde=True)
plt.title("Price distribution (raw)")
plt.show()

plt.figure(figsize=(12,4))
sns.histplot(np.log1p(df["Price"]), bins=60, kde=True)
plt.title("log1p(Price) distribution")
plt.show()

plt.figure(figsize=(12,4))
sns.boxplot(x=df["Price"])
plt.title("Price boxplot (outliers visible)")
plt.show()

# Some categorical exploration if columns exist
cat_candidates = ["city","room_type","property_type","bed_type","cancellation_policy","cleaning_fee"]
cat_cols_eda = [c for c in cat_candidates if c in df.columns]

for c in cat_cols_eda:
    plt.figure(figsize=(12,4))
    sns.countplot(data=df, x=c, order=df[c].value_counts().head(10).index)
    plt.title(f"Top values of {c}")
    plt.xticks(rotation=25)
    plt.show()

# Numeric correlation
num_cols_all = df.select_dtypes(include=[np.number]).columns.tolist()
corr = df[num_cols_all].corr(numeric_only=True)

plt.figure(figsize=(12,10))
sns.heatmap(corr, cmap="viridis", center=0)
plt.title("Correlation heatmap (numeric)")
plt.show()

# Top correlations with Price
if "Price" in corr.columns:
    display(corr["Price"].sort_values(ascending=False).head(20).to_frame("corr_with_Price"))

# ============================================================
# 6) FEATURE ENGINEERING: luxury_count + luxury_items from amenities
# ============================================================
LUXURY_KEYWORDS = [
    "Pool","Hot tub","Gym","View","Doorman","Washer","Dryer","Parking",
    "Elevator","Patio","Balcony","Air conditioning","Heating","Breakfast"
]

def clean_amenities(s):
    if pd.isna(s):
        return ""
    s = str(s)
    return re.sub(r'[{}\[\]"]', "", s)

def count_luxury_items(s):
    s = s.lower()
    return sum(1 for k in LUXURY_KEYWORDS if k.lower() in s)

def list_luxury_items(s, topk=10):
    s = s.lower()
    found = [k for k in LUXURY_KEYWORDS if k.lower() in s]
    if not found:
        return "Standard features"
    return ", ".join(found[:topk])

if "amenities" in df.columns:
    df["amenities_clean"] = df["amenities"].apply(clean_amenities)
    df["luxury_count"] = df["amenities_clean"].apply(count_luxury_items)
    df["luxury_items"] = df["amenities_clean"].apply(list_luxury_items)
else:
    df["luxury_count"] = 0
    df["luxury_items"] = "Standard features"

# Visualize luxury_count
plt.figure(figsize=(12,4))
sns.histplot(df["luxury_count"], bins=30, kde=True)
plt.title("luxury_count distribution")
plt.show()

plt.figure(figsize=(6,5))
sns.scatterplot(data=df.sample(min(8000, len(df)), random_state=RANDOM_SEED),
                x="luxury_count", y="Price", alpha=0.25)
plt.title("Price vs luxury_count")
plt.yscale("log")
plt.show()

# ============================================================
# 7) SELECT FEATURES (ROBUST TO COLUMN AVAILABILITY)
# ============================================================
num_candidates = [
    "accommodates","bathrooms","bedrooms","beds","review_scores_rating",
    "number_of_reviews","latitude","longitude","luxury_count"
]
cat_candidates = [
    "city","room_type","property_type","bed_type","cancellation_policy",
    "cleaning_fee","instant_bookable","host_identity_verified","host_has_profile_pic",
    "neighbourhood","zipcode"
]

num_cols = [c for c in num_candidates if c in df.columns]
cat_cols = [c for c in cat_candidates if c in df.columns]

print("Numeric features:", num_cols)
print("Categorical features:", cat_cols)

keep_cols = num_cols + cat_cols + ["Price","Category","luxury_items"]
df_model = df[keep_cols].copy()

# ============================================================
# 8) SPLIT: 70/15/15 (STRATIFY BY CATEGORY)
# ============================================================
train_df, temp_df = train_test_split(
    df_model, test_size=(1-TRAIN_FRAC), random_state=RANDOM_SEED, stratify=df_model["Category"]
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.50, random_state=RANDOM_SEED, stratify=temp_df["Category"]
)

print(f"\n Train: {train_df.shape}, Val: {val_df.shape}, Test: {test_df.shape}")

# ============================================================
# 9) PREPROCESSING:
# - Missing values (median numeric, most_frequent categorical)
# - OneHot for categorical
# - Normalize numeric using mean/std (StandardScaler)
# ============================================================
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler(with_mean=True, with_std=True))
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_cols),
        ("cat", categorical_transformer, cat_cols),
    ],
    remainder="drop"
)

# Fit ONLY on train
X_train = preprocess.fit_transform(train_df)
X_val   = preprocess.transform(val_df)
X_test  = preprocess.transform(test_df)

y_train = train_df["Price"].values.astype(np.float32)
y_val   = val_df["Price"].values.astype(np.float32)
y_test  = test_df["Price"].values.astype(np.float32)

y_cat_train = train_df["Category"].values
y_cat_val   = val_df["Category"].values
y_cat_test  = test_df["Category"].values

print(" Encoded shapes:", X_train.shape, X_val.shape, X_test.shape)

# Visualize normalized numeric (train)
n_num = len(num_cols)
if n_num > 0:
    train_num_scaled = X_train[:, :n_num]
    plt.figure(figsize=(12,4))
    for i in range(min(n_num, 6)):
        sns.kdeplot(train_num_scaled[:, i], label=num_cols[i])
    plt.title("Scaled numeric features (train) — should be ~ mean 0, std 1")
    plt.legend()
    plt.show()

# ============================================================
# 10) TF DATASETS
# ============================================================
BATCH_SIZE = 256

train_ds = tf.data.Dataset.from_tensor_slices((X_train.astype(np.float32), y_train)).shuffle(20000, seed=RANDOM_SEED).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
val_ds   = tf.data.Dataset.from_tensor_slices((X_val.astype(np.float32), y_val)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
test_ds  = tf.data.Dataset.from_tensor_slices((X_test.astype(np.float32), y_test)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# ============================================================
# 11) BEST DNN REGRESSION MODEL (FEEDFORWARD + BN + DROPOUT)
# ============================================================
tf.keras.utils.set_random_seed(RANDOM_SEED)

def acc_within_pct_np(y_true, y_pred, pct=0.10):
    rel_err = np.abs(y_pred - y_true) / np.maximum(np.abs(y_true), 1.0)
    return np.mean(rel_err <= pct)

def acc_within_pct_tf(y_true, y_pred, pct=0.10):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    rel_err = tf.abs(y_pred - y_true) / tf.maximum(tf.abs(y_true), 1.0)
    return tf.reduce_mean(tf.cast(rel_err <= pct, tf.float32))

class AccWithin10(tf.keras.metrics.Metric):
    def __init__(self, name="acc_within_10pct", **kwargs):
        super().__init__(name=name, **kwargs)
        self.total = self.add_weight(name="total", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")
    def update_state(self, y_true, y_pred, sample_weight=None):
        val = acc_within_pct_tf(y_true, y_pred, 0.10)
        self.total.assign_add(val)
        self.count.assign_add(1.0)
    def result(self):
        return tf.math.divide_no_nan(self.total, self.count)
    def reset_state(self):
        self.total.assign(0.0); self.count.assign(0.0)

class AccWithin20(tf.keras.metrics.Metric):
    def __init__(self, name="acc_within_20pct", **kwargs):
        super().__init__(name=name, **kwargs)
        self.total = self.add_weight(name="total", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")
    def update_state(self, y_true, y_pred, sample_weight=None):
        val = acc_within_pct_tf(y_true, y_pred, 0.20)
        self.total.assign_add(val)
        self.count.assign_add(1.0)
    def result(self):
        return tf.math.divide_no_nan(self.total, self.count)
    def reset_state(self):
        self.total.assign(0.0); self.count.assign(0.0)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1],)),

    tf.keras.layers.Dense(512, activation="relu"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.30),

    tf.keras.layers.Dense(256, activation="relu"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.25),

    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.20),

    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(1, activation="linear")
])

# Huber is often stronger than MAE alone, but still robust:
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=tf.keras.losses.Huber(delta=50.0),
    metrics=[
        tf.keras.metrics.MeanAbsoluteError(name="mae"),
        tf.keras.metrics.MeanAbsolutePercentageError(name="mape"),
        AccWithin10(),
        AccWithin20()
    ]
)

callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6)
]

print("\n Training regression NN (verbose=1)...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=60,
    verbose=1,
    callbacks=callbacks
)

# ============================================================
# 12) TRAINING VISUALIZATION (LOSS + METRICS)
# ============================================================
hist = pd.DataFrame(history.history)

plt.figure(figsize=(12,4))
plt.plot(hist["loss"], label="train_loss(Huber)")
plt.plot(hist["val_loss"], label="val_loss(Huber)")
plt.title("Loss vs Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.legend()
plt.show()

for m in ["mae","mape","acc_within_10pct","acc_within_20pct"]:
    if m in hist.columns:
        plt.figure(figsize=(12,4))
        plt.plot(hist[m], label=f"train_{m}")
        plt.plot(hist[f"val_{m}"], label=f"val_{m}")
        plt.title(f"{m} vs Epochs")
        plt.xlabel("Epoch")
        plt.grid(True)
        plt.legend()
        plt.show()

# ============================================================
# 13) TEST EVALUATION (RMSE FIXED WITHOUT squared=False)
# ============================================================
y_pred = model.predict(X_test.astype(np.float32), batch_size=1024).flatten()

mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)
mape = np.mean(np.abs((y_pred - y_test) / np.maximum(np.abs(y_test), 1))) * 100
acc10 = acc_within_pct_np(y_test, y_pred, 0.10) * 100
acc20 = acc_within_pct_np(y_test, y_pred, 0.20) * 100

print("\n================ TEST REGRESSION REPORT ================")
print(f"MAE:  {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²:   {r2:.3f}")
print(f"MAPE: {mape:.2f}%")
print(f"Accuracy@10%: {acc10:.2f}%")
print(f"Accuracy@20%: {acc20:.2f}%")
print("========================================================")

# ============================================================
# 14) REGRESSION GRAPHS (LOTS)
# ============================================================
# Pred vs True
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred, alpha=0.3, s=10)
mn, mx = np.min(y_test), np.max(y_test)
plt.plot([mn, mx], [mn, mx], "r--", label="Perfect fit")
plt.title("Predicted vs True Price")
plt.xlabel("True Price")
plt.ylabel("Predicted Price")
plt.grid(True)
plt.legend()
plt.show()

# Residuals
residuals = y_pred - y_test

plt.figure(figsize=(12,4))
sns.histplot(residuals, bins=60, kde=True)
plt.title("Residuals distribution (Pred - True)")
plt.xlabel("Residual")
plt.grid(True)
plt.show()

plt.figure(figsize=(12,4))
plt.scatter(y_test, residuals, alpha=0.25, s=10)
plt.axhline(0, linestyle="--")
plt.title("Residuals vs True Price")
plt.xlabel("True Price")
plt.ylabel("Residual (Pred-True)")
plt.grid(True)
plt.show()

abs_err = np.abs(residuals)
plt.figure(figsize=(12,4))
sns.histplot(abs_err, bins=60, kde=True)
plt.title("Absolute Error distribution")
plt.xlabel("|Pred-True|")
plt.grid(True)
plt.show()

plt.figure(figsize=(12,4))
plt.scatter(y_test, abs_err, alpha=0.25, s=10)
plt.title("Absolute Error vs True Price")
plt.xlabel("True Price")
plt.ylabel("|Error|")
plt.grid(True)
plt.show()

# ============================================================
# 15) TEST CASES TABLE (TRUE vs PRED + ERROR)
# ============================================================
test_cases = test_df.copy()
test_cases["Pred_Price"] = y_pred
test_cases["Abs_Error"] = np.abs(test_cases["Pred_Price"] - test_cases["Price"])
test_cases["Pct_Error_%"] = (test_cases["Abs_Error"] / np.maximum(test_cases["Price"], 1) * 100)

display(
    test_cases[["Price","Pred_Price","Abs_Error","Pct_Error_%","Category","luxury_items"] +
               [c for c in ["city","room_type","property_type","accommodates","bedrooms","bathrooms","review_scores_rating","luxury_count"] if c in test_cases.columns]]
    .sort_values("Abs_Error", ascending=True)
    .head(25)
)

# ============================================================
# 16) CONFUSION MATRIX (PRICE -> CATEGORY)
# ============================================================
nn_pred_cat = np.array([get_price_category(p) for p in y_pred])

cm = confusion_matrix(y_cat_test, nn_pred_cat, labels=["Low","Medium","High"])
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=["Low","Medium","High"], yticklabels=["Low","Medium","High"])
plt.title("Confusion Matrix (NN price→category)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

print("\nClassification report (NN mapped categories):")
print(classification_report(y_cat_test, nn_pred_cat, digits=3))

nn_cat_acc = accuracy_score(y_cat_test, nn_pred_cat) * 100
print(f" NN category accuracy (from regression): {nn_cat_acc:.2f}%")

# ============================================================
# 17) LLM FEW-SHOT CLASSIFICATION (LOW/MED/HIGH)
# ============================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print("\n Loading LLM:", LLM_MODEL_NAME, "| device:", device)

tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
llm = AutoModelForSeq2SeqLM.from_pretrained(LLM_MODEL_NAME).to(device)

few_shot = """
You are classifying Airbnb listings into price levels: Low, Medium, High.

Examples:
Example 1:
Input: city=NYC; room_type=Shared room; property_type=Apartment; accommodates=1; bedrooms=0; bathrooms=1; review_scores_rating=85; luxury_count=0; cleaning_fee=False
Label: Low

Example 2:
Input: city=LA; room_type=Entire home/apt; property_type=House; accommodates=4; bedrooms=2; bathrooms=2; review_scores_rating=93; luxury_count=3; cleaning_fee=True
Label: Medium

Example 3:
Input: city=DC; room_type=Entire home/apt; property_type=House; accommodates=8; bedrooms=4; bathrooms=3; review_scores_rating=98; luxury_count=6; cleaning_fee=True
Label: High

Now classify the next input. Answer with exactly one word: Low OR Medium OR High.
Input:
"""

def row_to_compact_features(row):
    parts = []
    for c in ["city","room_type","property_type","bed_type","cancellation_policy"]:
        if c in row.index:
            parts.append(f"{c}={str(row[c])}")
    for c in ["accommodates","bedrooms","bathrooms","beds","review_scores_rating","number_of_reviews","luxury_count"]:
        if c in row.index:
            parts.append(f"{c}={row[c]}")
    for c in ["cleaning_fee","instant_bookable","host_identity_verified","host_has_profile_pic"]:
        if c in row.index:
            parts.append(f"{c}={str(row[c])}")
    return "; ".join(parts)

def llm_classify(row):
    prompt = few_shot + row_to_compact_features(row) + "\nLabel:"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
    out = llm.generate(**inputs, max_new_tokens=3, temperature=0.1)
    pred = tokenizer.decode(out[0], skip_special_tokens=True).strip().replace(".", "").strip()
    if pred.lower().startswith("low"): return "Low"
    if pred.lower().startswith("med"): return "Medium"
    if pred.lower().startswith("high"): return "High"
    return pred

eval_n = min(LLM_EVAL_SAMPLES, len(test_df))
eval_rows = test_df.sample(eval_n, random_state=RANDOM_SEED).copy()

llm_preds = []
for _, r in eval_rows.iterrows():
    llm_preds.append(llm_classify(r))

llm_acc = accuracy_score(eval_rows["Category"].values, llm_preds) * 100
print(f"\n LLM few-shot accuracy (on {eval_n} test rows): {llm_acc:.2f}%")
print(f" NN category accuracy (full test, from regression): {nn_cat_acc:.2f}%")

cm_llm = confusion_matrix(eval_rows["Category"].values, llm_preds, labels=["Low","Medium","High"])
plt.figure(figsize=(6,5))
sns.heatmap(cm_llm, annot=True, fmt="d", xticklabels=["Low","Medium","High"], yticklabels=["Low","Medium","High"])
plt.title("Confusion Matrix (LLM few-shot)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

# ============================================================
# 18) LLM “CONVINCING” REASONS FOR PREDICTED PRICE requirement
# Uses important features and luxury_items
# ============================================================
def llm_reason_price(row, pred_price, true_price=None):
    feature_lines = []
    for c in (num_cols + cat_cols):
        if c in row.index:
            feature_lines.append(f"- {c}: {row[c]}")
    feature_lines.append(f"- luxury_items: {row.get('luxury_items','Standard features')}")
    feature_blob = "\n".join(feature_lines[:80])

    prompt = f"""
You are an expert Airbnb pricing analyst.
Your job: justify a predicted nightly price using the listing features.

Requirements:
- Be convincing and specific (not generic).
- Mention the most important features FIRST:
  city/location, room_type, property_type, accommodates, bedrooms, bathrooms,
  review_scores_rating, luxury_count/luxury_items, cancellation_policy, cleaning_fee.
- Use this structure:
  1) Predicted price: $X
  2) Top drivers (bullets)
  3) Secondary factors (bullets)
  4) Risk/uncertainty (bullets, 1-3 items)

Listing features:
{feature_blob}

Predicted price: ${pred_price:.0f}
"""
    if true_price is not None:
        prompt += f"\nTrue price (for evaluation only): ${true_price:.0f}\n"

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
    out = llm.generate(**inputs, max_new_tokens=240, temperature=0.35, repetition_penalty=1.15)
    return tokenizer.decode(out[0], skip_special_tokens=True).strip()

# Show a demo of 8 test rows with reasons
demo = test_df.copy()
demo["Pred_Price"] = y_pred
demo["NN_Category"] = nn_pred_cat

demo_samples = demo.sample(8, random_state=RANDOM_SEED).copy()
reasons = []
for idx, row in demo_samples.iterrows():
    reasons.append(llm_reason_price(row, row["Pred_Price"], true_price=row["Price"]))
demo_samples["LLM_Reasons_for_Price"] = reasons

cols_show = ["Price","Pred_Price","Category","NN_Category","luxury_items"] + \
            [c for c in ["city","room_type","property_type","accommodates","bedrooms","bathrooms","review_scores_rating","luxury_count","cleaning_fee"] if c in demo_samples.columns]

display(demo_samples[cols_show])
display(demo_samples[["LLM_Reasons_for_Price"]])


