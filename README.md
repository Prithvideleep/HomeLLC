# Housing Price Prediction

This project forecasts the Home Price Index (HPI) using machine learning and time series models, including Gradient Boosting, SARIMAX, and Prophet. The code loads historical housing and economic data, performs feature engineering, trains models, and generates forecasts with visualizations.

## Features

- Data preprocessing and feature engineering
- Gradient Boosting Regressor for regression
- SARIMAX and Prophet for time series forecasting
- Backtesting and future forecasting
- Visualization of actual and predicted HPI

## Setup

1. **Clone the repository** (if you haven't already):

    ```bash
    git clone <repository-url>
    cd AssignmentforHomeLLC
    ```

2. **Install dependencies**:

    Make sure you have Python 3.8 or newer installed. Then, install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

3. **Prepare the data**:

    Place your `final_dataset.csv` file inside a folder named `data` in the project directory:

    ```
    AssignmentforHomeLLC/
    ├── data/
    │   └── final_dataset.csv
    ├── housing_prediction.py
    └── requirements.txt
    ```

## Usage

To run the housing price prediction script and generate forecasts and plots, use:

```bash
python housing_prediction.py