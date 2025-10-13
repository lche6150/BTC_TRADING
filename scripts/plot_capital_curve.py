import re
from pathlib import Path
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt

log_path = Path('results/visualizations/trading_log_2024.txt')
lines = log_path.read_text().splitlines()
records = []
last_exit_time = None
year = 2024
exit_pattern = re.compile(r"Exit \d+:\s+(\d{2}/\d{2}) (\d{2}:\d{2})")
net_pattern = re.compile(r"Net P&L \(all fees\):\s+\$([+-]?[0-9.,]+)")

for line in lines:
    line = line.rstrip()
    exit_match = exit_pattern.search(line)
    if exit_match:
        month_day, time_str = exit_match.groups()
        dt = datetime.strptime(f"{year} {month_day} {time_str}", "%Y %m/%d %H:%M")
        if last_exit_time and dt < last_exit_time:
            year += 1
            dt = datetime.strptime(f"{year} {month_day} {time_str}", "%Y %m/%d %H:%M")
        last_exit_time = dt
        continue
    net_match = net_pattern.search(line)
    if net_match and last_exit_time:
        pnl = float(net_match.group(1).replace(',', ''))
        records.append({'timestamp': last_exit_time, 'net_pnl': pnl})
        last_exit_time = None

if not records:
    raise SystemExit('No trade records parsed from log')

df = pd.DataFrame(records).sort_values('timestamp').reset_index(drop=True)
initial_capital = 20000.0
df['capital'] = initial_capital + df['net_pnl'].cumsum()

out_dir = Path('results/visualizations')
out_dir.mkdir(parents=True, exist_ok=True)
plot_path = out_dir / 'capital_curve_2024.png'

plt.figure(figsize=(12, 6))
plt.plot(df['timestamp'], df['capital'], label='Remaining Capital', color='tab:blue')
plt.title('Capital Curve – 2024 Backtest')
plt.xlabel('Date')
plt.ylabel('Capital (USD)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.legend()
plt.savefig(plot_path)

csv_path = out_dir / 'capital_curve_2024.csv'
df[['timestamp', 'capital']].to_csv(csv_path, index=False)

print(f'Capital curve saved to {plot_path}')
print(f'Capital series exported to {csv_path}')
