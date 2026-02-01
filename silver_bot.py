import yfinance as yf
import pandas as pd
import numpy as np
import os
import requests
from hmmlearn.hmm import GaussianHMM
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# 1. FETCH DATA
tickers = ["SI=F", "GC=F"]
df_raw = yf.download(tickers, start="2016-01-01", interval="1d")['Close']
df = pd.DataFrame(df_raw['SI=F']).rename(columns={'SI=F': 'Close'})

# 2. FEATURE ENGINEERING
df['GS_Ratio'] = df_raw['GC=F'] / df_raw['SI=F']
df['RSI'] = 100 - (100 / (1 + (df['Close'].diff().where(df['Close'].diff() > 0, 0).rolling(14).mean() / 
                              -df['Close'].diff().where(df['Close'].diff() < 0, 0).rolling(14).mean())))
df['Volatility'] = df['Close'].pct_change().rolling(20).std()
df['MA_Diff'] = df['Close'].rolling(5).mean() - df['Close'].rolling(20).mean()
df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
df.dropna(inplace=True)

# 3. SELF-LEARNING & VALIDATION
features = ['GS_Ratio', 'RSI', 'Volatility', 'MA_Diff']
train_df = df.copy()

# Level 1: Regime Detection
hmm = GaussianHMM(n_components=3, n_iter=1000, tol=0.01).fit(train_df[['Volatility']])
train_df['Regime'] = hmm.predict(train_df[['Volatility']])

# Level 2: Training with Historical Accuracy Check
X = train_df[features + ['Regime']]
y = train_df['Target']

# Split to see how we performed on recent history (last 20% of data)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
clf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
clf.fit(X_train, y_train)

# Calculate Accuracy on the test set
y_pred = clf.predict(X_test)
hist_accuracy = accuracy_score(y_test, y_pred)

# 4. FINAL PREDICTION (Using the latest data point)
latest_row = train_df.iloc[-1:]
regime = hmm.predict(latest_row[['Volatility']])[0]
final_features = latest_row[features].copy()
final_features['Regime'] = regime

prediction = clf.predict(final_features)[0]
probs = clf.predict_proba(final_features)[0]
confidence = probs[1] if prediction == 1 else probs[0]

signal = "🚀 BUY" if prediction == 1 else "📉 SELL"

# 5. TELEGRAM NOTIFICATION
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

message = (
    f"🥈 *Silver Daily Signal*\n\n"
    f"State: {('STABLE' if regime == 0 else 'VOLATILE' if regime == 1 else 'EXTREME')}\n"
    f"Signal: *{signal}*\n"
    f"🎯 Confidence: {confidence:.1%}\n"
    f"📊 Past Accuracy: {hist_accuracy:.1%}\n\n"
    f"Price: ${latest_row['Close'].values[0]:.2f}"
)

if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    try:
        response = requests.post(url, json=payload)
        print("Telegram Response:", response.text)
    except Exception as e:
        print(f"Error sending Telegram: {e}")
else:
    print("Secrets missing. Telegram message not sent.")
    print(message)
