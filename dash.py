import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
from sklearn.metrics import classification_report

final_model = joblib.load('final_model.pkl')
x_test = np.load('x_test.npy')
y_test = np.load('y_test.npy')
idx_test = np.load('idx_test.npy')
df = pd.read_csv('df.csv')

st.sidebar.header("Betting Parameters")
threshold_margin = st.sidebar.slider("Betting Margin", 0.01, 0.20, 0.05, step=0.01)
stake = st.sidebar.number_input("Stake per Bet", min_value=1.0, max_value=10000.0, value=1.0, step=1.0)


bet_results = []

probs = final_model.predict(x_test)
y_true = np.argmax(y_test, axis=1)

features = ['HST', 'AST', 'HC', 'AC', 'Home_xG', 'Away_xG',
            'ShotDiff', 'CornerDiff', 'FoulDiff', 'xGDiff',
            'Norm_ShotDiff', 'Norm_CornerDiff', 'Norm_FoulDiff', 'Norm_xGDiff']

X_test_betting = pd.DataFrame(x_test, columns=features)
X_test_betting["B365H"] = df.iloc[idx_test]["B365H"].values
X_test_betting["B365D"] = df.iloc[idx_test]["B365D"].values
X_test_betting["B365A"] = df.iloc[idx_test]["B365A"].values

X_test_betting["B365H_prob"] = 1 / X_test_betting["B365H"]
X_test_betting["B365D_prob"] = 1 / X_test_betting["B365D"]
X_test_betting["B365A_prob"] = 1 / X_test_betting["B365A"]
total = X_test_betting[["B365H_prob", "B365D_prob", "B365A_prob"]].sum(axis=1)
X_test_betting["B365H_prob"] /= total
X_test_betting["B365D_prob"] /= total
X_test_betting["B365A_prob"] /= total

for i in range(len(X_test_betting)):
    game = X_test_betting.iloc[i]
    true_result = y_true[i]
    prob_pred = probs[i]
    odds = [1 / game['B365A_prob'], 1 / game['B365D_prob'], 1 / game['B365H_prob']]
    implied_probs = [game['B365A_prob'], game['B365D_prob'], game['B365H_prob']]

    for outcome in range(3):
        if prob_pred[outcome] > implied_probs[outcome] + threshold_margin:
            win = int(true_result == outcome)
            profit = (odds[outcome] - 1) * stake if win else -stake
            bet_results.append({
                'game_index': i,
                'outcome': outcome,
                'bet_on': ['A', 'D', 'H'][outcome],
                'prob_pred': prob_pred[outcome],
                'implied_prob': implied_probs[outcome],
                'win': win,
                'profit': profit
            })

bets_df = pd.DataFrame(bet_results)
total_bets = len(bets_df)
total_profit = bets_df['profit'].sum()
roi = total_profit / (total_bets * stake) if total_bets > 0 else 0
win_rate = bets_df['win'].mean() if total_bets > 0 else 0
bets_df['cumulative_profit'] = bets_df['profit'].cumsum()

st.title("Betting Simulation using Neural Network Dashboard")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Bets", total_bets)
col2.metric("Total Profit", f"${total_profit:.2f}")
col3.metric("ROI", f"{roi * 100:.2f}%")
col4.metric("Win Rate", f"{win_rate * 100:.2f}%")

st.subheader("Cumulative Profit")
fig = px.line(bets_df, y='cumulative_profit', title='Cumulative Profit ',
              labels={'index': 'Number of Bets', 'cumulative_profit': 'Cumulative Profit'})
st.plotly_chart(fig, use_container_width=True)

st.subheader("Betting Results")
st.dataframe(bets_df)

outcome_filter = st.multiselect(
    'Filter by Bet Outcome:',
    options=['H', 'D', 'A'],
    default=['H', 'D', 'A']
)

filtered_df = bets_df[bets_df['bet_on'].isin(outcome_filter)]
st.write(filtered_df)

