# Sales Forecast Insights (Monthly)

## Executive Summary
We forecasted monthly sales using SARIMA and Prophet with an 80/20 time split. The chosen model achieved the best error metrics on the holdout test set and was used to forecast the next 6 and 12 months.

## Key Patterns Observed
- **Strong seasonality:** Sales consistently peak in **November–December**.
- **Upward trend:** Overall sales trend increases over time (rolling average rises).

## Operational Recommendations
1. **Inventory planning**
   - Increase inventory ahead of Q4 demand (Oct–Dec).
   - Prioritize fast-moving categories and regions for replenishment buffers.

2. **Staffing**
   - Plan higher staffing levels in Nov–Dec due to recurring peak demand.
   - Align warehouse and customer service scheduling to the forecasted peak.

3. **Finance & budgeting**
   - Use the 12-month forecast to set revenue targets and purchasing budgets.
   - Use confidence intervals for risk-aware planning (best/worst-case scenarios).

## Model Notes (Interview Talking Points)
- SARIMA is a strong statistical baseline for stable seasonality.
- Prophet is business-friendly, handles seasonality automatically, and is easier to extend with holidays/promotions.