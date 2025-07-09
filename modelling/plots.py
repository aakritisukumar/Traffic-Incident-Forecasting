import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from preprocessing import load_and_preprocess_data

df, monthly_data, annual_data = load_and_preprocess_data()

def plot_monthly_trends():
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(monthly_data.index, monthly_data['WERT'], marker='o', label='Monthly Accidents')
    ax.set_title('Monthly Accident Trends')
    ax.set_xlabel('Date')
    ax.set_ylabel('Number of Accidents')
    ax.grid(True)
    ax.legend()
    return fig

def plot_annual_totals():
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(annual_data.index.astype(str), annual_data['WERT'], color='orange', label='Annual Totals')
    ax.set_title('Annual Accident Totals')
    ax.set_xlabel('Year')
    ax.set_ylabel('Number of Accidents')
    ax.grid(axis='y')
    ax.legend()
    return fig

def plot_trend_line():
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(x=monthly_data.index, y=monthly_data['WERT'], ax=ax)
    ax.set_title('Trend of Alcohol-Related Accidents Over Time (Monthly)')
    ax.set_xlabel('Date')
    ax.set_ylabel('Number of Accidents')
    return fig

def plot_seasonal_decomposition(period=12):
    result = seasonal_decompose(monthly_data['WERT'], model='additive', period=period)
    fig, axes = plt.subplots(4, 1, figsize=(12, 8))  # Create subplots manually
    result.observed.plot(ax=axes[0], title='Observed', color='blue')
    result.trend.plot(ax=axes[1], title='Trend', color='orange')
    result.seasonal.plot(ax=axes[2], title='Seasonal', color='green')
    result.resid.plot(ax=axes[3], title='Residual', color='red')
    fig.tight_layout()
    return fig

def plot_acf_pacf(lags=24):
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    plot_acf(monthly_data['WERT'], ax=axes[0], lags=lags)
    axes[0].set_title('Autocorrelation Function')
    plot_pacf(monthly_data['WERT'], ax=axes[1], lags=lags)
    axes[1].set_title('Partial Autocorrelation Function')
    fig.tight_layout()
    return fig
