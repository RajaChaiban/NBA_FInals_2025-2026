🏀 I built a machine learning model to predict the 2026 NBA Finals: Spurs vs Knicks.

📊 THE RESULT — a near coin flip:
• Spurs win series: 50.2%
• Knicks win series: 49.8%
• Model accuracy on unseen games: 68.6%

🧠 HOW IT WORKS
3 seasons of official NBA stats (2023-26, ~3,900 games via nba_api) → an XGBoost + Logistic Regression ensemble → a Monte Carlo simulation of the best-of-7.

🔑 THE VARIABLES
Net/offensive/defensive rating, the Four Factors (shooting, turnovers, rebounding, free throws), an Elo power rating, rest, home court, and head-to-head — all computed pre-game, one team minus the other.

⚠️ WHAT IT CAN'T SEE
The biggest factor isn't in any box score: INJURIES. Wembanyama's health (he missed half of last season with a blood clot) could decide the whole series. Plus momentum, coaching adjustments, and one hot shooting night.

Good ML doesn't pretend to know more than it does — and the honest answer here is "too close to call." 🐍

Built in Python with nba_api, scikit-learn & XGBoost.

#MachineLearning #DataScience #SportsAnalytics #NBA #Python
