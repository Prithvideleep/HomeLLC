# US Home Price Prediction Model

This project builds a data science model to analyze and predict how key economic factors influence US home prices over the last 20 years. The S&P Case-Schiller Home Price Index is used as a proxy for home prices, and publicly available data is sourced for the analysis.

## Project Overview

The goal of this project is to:
1. Collect and preprocess data for key economic factors that influence US home prices.
2. Build a linear regression model to predict home prices based on these factors.
3. Evaluate the model's performance and visualize the comparison between actual and predicted values.

## Data Sources

The data is sourced from the [Federal Reserve Economic Data (FRED)](https://fred.stlouisfed.org/). The following economic factors are included:
- **Home Price Index**: S&P Case-Schiller Home Price Index (`CSUSHPISA`)
- **Mortgage Rate**: 30-Year Fixed Mortgage Rate (`MORTGAGE30US`)
- **Unemployment Rate**: (`UNRATE`)
- **Treasury Yield**: 10-Year Treasury Yield (`GS10`)
- **GDP**: Gross Domestic Product (`GDP`)
- **Median Income**: Median Household Income (`MEHOINUSA672N`)
- **CPI**: Consumer Price Index (`CPIAUCSL`)
- **Housing Starts**: (`HOUST`)
- **Population**: (`POP`)

## Project Setup

1. **Environment Setup**:
   - Install Python 3.7 or later from [python.org](https://www.python.org/downloads/)
   - Create a virtual environment (recommended):
     ```bash
     python -m venv venv
     .\venv\Scripts\activate
     ```

2. **Install Dependencies**:
   ```bash
   pip install fredapi pandas matplotlib seaborn scikit-learn
   ```

3. **FRED API Key**:
   - Get your FRED API key from [FRED API](https://fred.stlouisfed.org/docs/api/api_key.html)
   - Create a `.env` file in the project root:
     ```bash
     FRED_API_KEY=your_api_key_here
     ```

## Project Structure

```
HOMELLC2.0/
│
├── data/
│   ├── raw/                  # Raw data from FRED
│   └── processed/            # Processed and cleaned datasets
│
├── notebooks/
│   └── home_price_model.ipynb # Main analysis notebook
│
├── output/
│   ├── final_dataset.csv     # Final processed dataset
│   └── feature_impact.csv    # Feature importance results
│
├── .env                      # Environment variables
├── requirements.txt          # Project dependencies
└── README.md                 # Project documentation
```

## Workflow

1. **Data Collection**:
   - Open `home_price_model.ipynb`
   - Run the data collection cells to fetch data from FRED
   - Raw data is saved in `data/raw/`

2. **Data Preprocessing**:
   - Handle missing values
   - Feature engineering
   - Data normalization
   - Save processed data to `data/processed/`

3. **Model Training**:
   - Split data into training and testing sets
   - Train linear regression model
   - Save model artifacts

4. **Model Evaluation**:
   - Calculate R² Score and MSE
   - Generate prediction vs actual plots
   - Export feature importance to `feature_impact.csv`

## Model Performance

The model's performance is evaluated using:
- R² Score: Indicates variance explanation (typical range: 0.7-0.9)
- Mean Squared Error (MSE): Measures prediction accuracy
- Feature importance analysis in `feature_impact.csv`

## Visualization Examples

The notebook generates several visualizations:
- Time series plots of economic factors
- Correlation heatmap
- Actual vs predicted home prices
- Feature importance bar chart

## Troubleshooting

Common issues and solutions:
1. FRED API errors: Verify API key in `.env` file
2. Missing data: Check date ranges in data collection
3. Memory issues: Reduce data size or use chunks

## Contributing

1. Fork the repository
2. Create a feature branch
3. Submit a pull request with detailed changes

## License

This project is for educational purposes and does not include a specific license.

## Contact

For questions or feedback, please reach out via the submission platform.