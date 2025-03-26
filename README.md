# S&P 500 Directional Prediction with ARIMA  
A trading strategy model predicting S&P 500 price movements using ARIMA time-series forecasting.  

## **Key Performance Metrics**  
- **Profit Factor**: 1.25 (Profitable)  
- **Direction Accuracy**: 57.3%  
- **Annualized Sharpe Ratio**: 1.25
- **Win Rate**: 55.4%  
- **Max Drawdown**: *(Needs recalculation)*  

## **How It Works**  
1. **Data**: 5 years of S&P 500 daily closes  
2. **Model**: ARIMA(2,0,1) with walk-forward validation  
3. **Execution**: Predicts next-day direction (long/short) with 5bps transaction costs  

## **Limitations**  
- Struggles with downturns (high false positives in bear markets)  
- Drawdown calculation needs fixing (currently unrealistic)  

## **Files**  
- `Price5yrs.m`: Main backtesting script  
- `HistoricalData_1742929324297.csv`: S&P 500 price data  

## **Next Steps**  
- Fix max drawdown calculation  
- Add volatility filters to improve win rate
- Experiment with ML models (LSTM, XGBoost) 
