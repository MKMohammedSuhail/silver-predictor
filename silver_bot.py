import yfinance as yf
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.ensemble import RandomForestClassifier
import os

# 1. DATA & TRAINING (Combined)
tickers = ["SI=F", "GC=F"]
df_raw = yf.download(tickers, start="2016-01-01", interval="1d")['Close']
df = pd.DataFrame(df_raw['SI=F']).rename(columns={'SI=F': 'Close'})
df['GS_Ratio'] = df_raw['GC=F'] / df_raw['SI=F']
df['RSI'] = 100 - (100 / (1 + (df['Close'].diff().where(df['Close'].diff() > 0, 0).rolling(14).mean() / 
                              -df['Close'].diff().where(df['Close'].diff() < 0, 0).rolling(14).mean())))
df['Volatility'] = df['Close'].pct_change().rolling(20).std()
df['MA_Diff'] = df['Close'].rolling(5).mean() - df['Close'].rolling(20).mean()
df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
df.dropna(inplace=True)

# Train fresh model every time
train_df = df.copy()
features = ['GS_Ratio', 'RSI', 'Volatility', 'MA_Diff']
hmm = GaussianHMM(n_components=3, n_iter=1000).fit(train_df[['Volatility']])
train_df['Regime'] = hmm.predict(train_df[['Volatility']])
clf = RandomForestClassifier(n_estimators=200, max_depth=10).fit(train_df[features + ['Regime']], train_df['Target'])

# 2. PREDICTION
latest_row = train_df.iloc[-1:]
regime = hmm.predict(latest_row[['Volatility']])[0]
final_features = latest_row[features].copy()
final_features['Regime'] = regime
prediction = clf.predict(final_features)[0]

# 3. OUTPUT
signal = "🚀 BUY" if prediction == 1 else "📉 SELL"
print(f"SIGNAL: {signal}")


# (Later we will add the Email/Telegram code here)
import requests

# Get secrets from GitHub
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

message = f"🥈 *Silver Daily Signal*\n\nState: {('STABLE' if regime == 0 else 'VOLATILE')}\nSignal: {signal}"

if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    requests.post(url, json=payload)
    print("Notification sent to Telegram!")
