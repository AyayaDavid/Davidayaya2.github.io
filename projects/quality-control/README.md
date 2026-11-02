# Manufacturing Quality Control Dashboard (SECOM)

Statistical process control (SPC) analysis of a real semiconductor production line by **David Ayaya**.

**Live dashboard:** https://ayayadavid.github.io/Davidayaya2.github.io/projects/quality-control/

## Questions
1. What is the line's yield, and is the failure rate stable over time?
2. Which sensor signals are most associated with failures?
3. Would an SPC chart on that signal have caught the problem?
4. How clean is the sensor data, and what should a monitoring plan include?

## Key findings
- 1,567 production runs, 590 sensors, **93.4% yield** (104 failures).
- The weekly **p-chart** shows the process was not stable: the week of Jul 28 broke the upper control limit, and the failure rate fell from **12.9%** (Jul 19 to Aug 24) to **4.1%** (Aug 25 onward).
- Sensor **S060** is the strongest signal. Against control limits set from the stable period, 47.4% of earlier runs were out of control versus 1.3% afterward. Runs outside the limits failed **17.6%** of the time versus 4.8% inside.
- Runs in the highest S060 quartile failed **15.1%** of the time versus 2.6% to 5.4% for the rest.
- Data checks: 116 sensors never change and 28 are more than half missing; 446 of 590 are usable.

## Recommendations
- Monitor S060 on a live I-MR chart with stable-period limits and alert on points outside them.
- Record what changed around Aug 25 in change control so the improvement is held.
- Remove constant and mostly-missing sensors from operator dashboards.
- Confirm root cause with process engineering before acting (observational data, anonymized sensors).

## Files
| File | What it is |
|---|---|
| `index.html` | Interactive dashboard (Chart.js) |
| `SECOM_Quality_Control_Analysis.ipynb` | Step-by-step Python notebook with charts |
| `analysis.py` | Script that produces `results.json` for the dashboard: `python analysis.py data results.json` |
| `data/` | Original SECOM data files |

## Data
SECOM dataset, UCI Machine Learning Repository (M. McCann and A. Johnston, 2008), licensed CC BY 4.0: https://archive.ics.uci.edu/dataset/179/secom

**Tools:** Python, Pandas, NumPy, Matplotlib, Chart.js
