import streamlit as st
import pandas as pd
from plotly import graph_objs as go
import plotly.express as px
from scipy.stats import linregress
from datetime import date
import yfinance as yf
import xlsxwriter
import io
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from time import time
from SinglePeriod import SinglePeriod
from classical_po import ClassicalPO
from scipy.optimize import minimize
from SinglePeriod import SinglePeriod
import optuna
from itertools import product
import dimod
import datetime
from dimod import quicksum, Integer, Binary
from dimod import ConstrainedQuadraticModel
from dwave.system import LeapHybridCQMSampler
#from tabulate import tabulate
from docplex.mp.model import Model
from qiskit_optimization.translators import from_docplex_mp
from qiskit_optimization.converters import QuadraticProgramToQubo
from dwave.samplers import SimulatedAnnealingSampler
import ml_collections
from statsmodels.tsa.filters.hp_filter import hpfilter
optuna.logging.set_verbosity(optuna.logging.WARNING)
#from alpha_vantage.timeseries import TimeSeries
from datetime import datetime, date, timedelta


##################### session-code
if "show_rebalancing_button" not in st.session_state:
    st.session_state["show_rebalancing_button"] = False

if "show_reset_button" not in st.session_state:
    st.session_state["show_reset_button"] = False








########################## Code for dynamic date within a timeframe
fixed_start_date = date(2023, 1, 9)
fixed_end_date = date(2024, 9, 25)








seed = 42
cfg = ml_collections.ConfigDict()

st.set_page_config(page_title="PilotProject", page_icon=":chart_with_upwards_trend:", layout="wide")
st.title("Pilot Project on Portfolio Optimization")

uploaded_files = st.file_uploader("Choose Excel files", type=["xlsx", "xls"], accept_multiple_files=True)
file_names = [uploaded_file.name for uploaded_file in uploaded_files]
selected_file = st.selectbox("Select a file", file_names)
select_benchmark = st.selectbox("Select the benchmark", options=['NIFTY 50', 'NIFTY 100', 'NIFTY 500', 'BSE 100', 'BSE 500'])
#end_date = st.date_input("Select end date", value=datetime(2024, 9, 25))
start_date = st.date_input("Select start date:", value=fixed_start_date, min_value=fixed_start_date, max_value=fixed_end_date)

end_date = st.date_input("Select an end date:", value=fixed_end_date, min_value=fixed_start_date, max_value=fixed_end_date)







#start_date = max(start_date, fixed_start_date)

## Dynamic end date
end_date = min(end_date, fixed_end_date)



total_budget = 1

def cleanName(name):
    if isinstance(name, str):
        return name.lower().replace("limited", "").replace("ltd", "").replace("ltd.", "").strip()
    return name



def getTempHoldingFile(df1, df2):

    mergedDf = pd.merge(left = df1,
                        right = df2,
                        on = "Company name",
                        how = "left",
                        indicator = True)
    
    unmatchedStocks = mergedDf.loc[mergedDf['_merge'] == 'left_only', 'Company name']

    mergedDf.drop(columns=['_merge'], inplace=True)

    return mergedDf, unmatchedStocks

def get_price(ticker, date, date_):
    """
    Fetches the stock price for a specific ticker on a given date.
    """
    try:
        stock_data = yf.download(ticker, start=date, end=date_, multi_level_index=False)['Close']
        #stock_data = stock_data['Close'].squeeze()
        # st.write(type(stock_data))
        # st.write(stock_data)
        if not stock_data.empty:
            return stock_data.iloc[0]
        else:
            return None
    except Exception as e:
        print(f"Error fetching price for {ticker} on {date}: {e}")
        return None

def fetch_prices(input_file, stock_column, start_date_column, start_date_jan, start_date_jan_, end_date_present, end_date_present_, output_file):
    """
    Reads an Excel file, fetches stock prices on start and end dates, 
    and writes the updated data to a new Excel file.
    """
    # Read the Excel file
    df = pd.read_excel(input_file)
    
    # Add new columns for prices
    start_prices = []
    end_prices = []
    
    for _, row in df.iterrows():
        ticker = row[stock_column]
        print(ticker)
        #start_date = row[start_date_column]
        
        # Fetch prices for start and end dates
        start_price = get_price(ticker, start_date_jan, start_date_jan_)
        end_price = get_price(ticker, end_date_present, end_date_present_)
        
        start_prices.append(start_price)
        end_prices.append(end_price)
    
    # Add the fetched prices to the DataFrame
    df['Price'] = start_prices
    df['CurrentSP'] = end_prices
    
    # Replace None values with empty strings
    df.fillna('', inplace=True)
    
    # Save the updated DataFrame to a new Excel file
    df.to_excel(output_file, index=False)
    print(f"Updated file with prices saved to '{output_file}'")

    return




#if st.button('Next'):
def next_button():
    if uploaded_files:
        for uploaded_file in uploaded_files:
            if uploaded_file.name == selected_file:

                holdingDf = pd.read_excel(uploaded_file)
                holdingDf["Company name"] = holdingDf["Company name"].apply(cleanName)

                global end_date
                global start_date

                if select_benchmark == 'NIFTY 50':

                    ## Merging the holding file and benchmark library
                    nifty50Benchmark = pd.read_excel("./benchmarks/nifty50Benchmark.xlsx")
                    nifty50Benchmark["Company name"] = nifty50Benchmark["Company name"].apply(cleanName)

                    tempHoldingFile, tempUnmatchedStocks = getTempHoldingFile(holdingDf, nifty50Benchmark)

                    tempHoldingFile.dropna(inplace=True)  # Remove rows with NaN values
                    tempHoldingFile['Date'] = pd.to_datetime(tempHoldingFile['Date'])
                    #start_date = tempHoldingFile['Date'].min()


                    #global end_date
                    end_date = pd.to_datetime(end_date)

                    tempHoldingFile.to_excel("tempHoldingFile.xlsx", index = False)
                    tempUnmatchedStocks.to_excel("tempUnmatchedStocks.xlsx", index = False)

                    
                    ## Fetching the price and CurrentSP
                    input_file = "tempHoldingFile.xlsx"  # Path to the input Excel file
                    stock_column = "SECURITY_ID"       # Column name for stock tickers
                    start_date_column = "Date" # Column name for start dates
                    # start_date_jan = "2023-01-09"
                    # start_date_jan_ = "2023-01-10"

                    #start_date_jan = start_date.date()
                    start_date_jan = start_date
                    start_date_jan_ = start_date_jan + timedelta(days=1)

                    end_date_present = "2024-12-31"
                    end_date_present_ = "2025-01-01"     # Column name for end dates
                    output_file = "tempHoldingFile.xlsx"  # Path for the output Excel file

                    fetch_prices(input_file, stock_column, start_date_column, start_date_jan, start_date_jan_, end_date_present, end_date_present_, output_file)


                    
                    # excel_file_nifty50 = pd.read_excel("./benchmarks/nifty50Benchmark.xlsx")
                    # excel_file_nifty50.to_csv("./benchmarks/nifty50Benchmark.csv", index = False)
                    # csv_file = "./benchmarks/nifty50Benchmark.csv"
                    # benchdata = yf.download("^NSEI", start = start_date, end = end_date)['Adj Close']
                    # sectorbenchmark = pd.read_csv('./benchmarks/nifty50SectorWeightages.csv')
                    # name = 'NIFTY 50'


                    ## get the sector weightages

                if select_benchmark == 'NIFTY 100':

                    ## Merging the holding file and benchmark library
                    nifty100Benchmark = pd.read_excel("./benchmarks/nifty100Benchmark.xlsx")
                    nifty100Benchmark["Company name"] = nifty100Benchmark["Company name"].apply(cleanName)

                    tempHoldingFile, tempUnmatchedStocks = getTempHoldingFile(holdingDf, nifty100Benchmark)

                    tempHoldingFile.dropna(inplace=True)  # Remove rows with NaN values
                    tempHoldingFile['Date'] = pd.to_datetime(tempHoldingFile['Date'])
                    #start_date = tempHoldingFile['Date'].min()
                    end_date = pd.to_datetime(end_date)

                    tempHoldingFile.to_excel("tempHoldingFile.xlsx", index = False)
                    tempUnmatchedStocks.to_excel("tempUnmatchedStocks.xlsx", index = False)

                    
                    ## Fetching the price and CurrentSP
                    input_file = "tempHoldingFile.xlsx"  # Path to the input Excel file
                    stock_column = "SECURITY_ID"       # Column name for stock tickers
                    start_date_column = "Date" # Column name for start dates
                    # start_date_jan = "2023-01-09"
                    # start_date_jan_ = "2023-01-10"

                    #start_date_jan = start_date.date()
                    start_date_jan = start_date
                    start_date_jan_ = start_date_jan + timedelta(days=1)


                    end_date_present = "2024-12-31"
                    end_date_present_ = "2025-01-01"     # Column name for end dates
                    output_file = "tempHoldingFile.xlsx"  # Path for the output Excel file

                    fetch_prices(input_file, stock_column, start_date_column, start_date_jan, start_date_jan_, end_date_present, end_date_present_, output_file)
                    


                if select_benchmark == 'NIFTY 500':

                    ## Merging the holding file and benchmark library
                    nifty500Benchmark = pd.read_excel("./benchmarks/nifty500Benchmark.xlsx")
                    nifty500Benchmark["Company name"] = nifty500Benchmark["Company name"].apply(cleanName)

                    tempHoldingFile, tempUnmatchedStocks = getTempHoldingFile(holdingDf, nifty500Benchmark)

                    tempHoldingFile.dropna(inplace=True)  # Remove rows with NaN values
                    tempHoldingFile['Date'] = pd.to_datetime(tempHoldingFile['Date'])
                    #start_date = tempHoldingFile['Date'].min()
                    end_date = pd.to_datetime(end_date)

                    tempHoldingFile.to_excel("tempHoldingFile.xlsx", index = False)
                    tempUnmatchedStocks.to_excel("tempUnmatchedStocks.xlsx", index = False)

                    
                    ## Fetching the price and CurrentSP
                    input_file = "tempHoldingFile.xlsx"  # Path to the input Excel file
                    stock_column = "SECURITY_ID"       # Column name for stock tickers
                    start_date_column = "Date" # Column name for start dates
                    # start_date_jan = "2023-01-09"
                    # start_date_jan_ = "2023-01-10"

                    #start_date_jan = start_date.date()
                    start_date_jan = start_date
                    start_date_jan_ = start_date_jan + timedelta(days=1)




                    end_date_present = "2024-12-31"
                    end_date_present_ = "2025-01-01"     # Column name for end dates
                    output_file = "tempHoldingFile.xlsx"  # Path for the output Excel file

                    fetch_prices(input_file, stock_column, start_date_column, start_date_jan, start_date_jan_, end_date_present, end_date_present_, output_file)

                

                if select_benchmark == 'SENSEX 50':

                    ## Merging the holding file and benchmark library
                    sensex50Benchmark = pd.read_excel("./benchmarks/sensex50Benchmark.xlsx")
                    sensex50Benchmark["Company name"] = sensex50Benchmark["Company name"].apply(cleanName)

                    tempHoldingFile, tempUnmatchedStocks = getTempHoldingFile(holdingDf, sensex50Benchmark)

                    tempHoldingFile.dropna(inplace=True)  # Remove rows with NaN values
                    tempHoldingFile['Date'] = pd.to_datetime(tempHoldingFile['Date'])
                    #start_date = tempHoldingFile['Date'].min()
                    end_date = pd.to_datetime(end_date)

                    tempHoldingFile.to_excel("tempHoldingFile.xlsx", index = False)
                    tempUnmatchedStocks.to_excel("tempUnmatchedStocks.xlsx", index = False)

                    
                    ## Fetching the price and CurrentSP
                    input_file = "tempHoldingFile.xlsx"  # Path to the input Excel file
                    stock_column = "SECURITY_ID"       # Column name for stock tickers
                    start_date_column = "Date" # Column name for start dates
                    # start_date_jan = "2023-01-09"
                    # start_date_jan_ = "2023-01-10"

                    #start_date_jan = start_date.date()
                    start_date_jan = start_date
                    start_date_jan_ = start_date_jan + timedelta(days=1)


                    end_date_present = "2024-12-31"
                    end_date_present_ = "2025-01-01"     # Column name for end dates
                    output_file = "tempHoldingFile.xlsx"  # Path for the output Excel file

                    fetch_prices(input_file, stock_column, start_date_column, start_date_jan, start_date_jan_, end_date_present, end_date_present_, output_file)
                


                if select_benchmark == 'BSE 100':

                    ## Merging the holding file and benchmark library
                    bse100Benchmark = pd.read_excel("./benchmarks/bse100Benchmark.xlsx")
                    bse100Benchmark["Company name"] = bse100Benchmark["Company name"].apply(cleanName)

                    tempHoldingFile, tempUnmatchedStocks = getTempHoldingFile(holdingDf, bse100Benchmark)

                    tempHoldingFile.dropna(inplace=True)  # Remove rows with NaN values
                    tempHoldingFile['Date'] = pd.to_datetime(tempHoldingFile['Date'])
                    #start_date = tempHoldingFile['Date'].min()
                    end_date = pd.to_datetime(end_date)

                    tempHoldingFile.to_excel("tempHoldingFile.xlsx", index = False)
                    tempUnmatchedStocks.to_excel("tempUnmatchedStocks.xlsx", index = False)

                    
                    ## Fetching the price and CurrentSP
                    input_file = "tempHoldingFile.xlsx"  # Path to the input Excel file
                    stock_column = "SECURITY_ID"       # Column name for stock tickers
                    start_date_column = "Date" # Column name for start dates
                    # start_date_jan = "2023-01-09"
                    # start_date_jan_ = "2023-01-10"

                    #start_date_jan = start_date.date()
                    start_date_jan = start_date
                    start_date_jan_ = start_date_jan + timedelta(days=1)


                    end_date_present = "2024-12-31"
                    end_date_present_ = "2025-01-01"     # Column name for end dates
                    output_file = "tempHoldingFile.xlsx"  # Path for the output Excel file

                    fetch_prices(input_file, stock_column, start_date_column, start_date_jan, start_date_jan_, end_date_present, end_date_present_, output_file)

                


                if select_benchmark == 'BSE 500':

                    ## Merging the holding file and benchmark library
                    bse500Benchmark = pd.read_excel("./benchmarks/bse500Benchmark.xlsx")
                    bse500Benchmark["Company name"] = bse500Benchmark["Company name"].apply(cleanName)

                    tempHoldingFile, tempUnmatchedStocks = getTempHoldingFile(holdingDf, bse500Benchmark)

                    tempHoldingFile.dropna(inplace=True)  # Remove rows with NaN values
                    tempHoldingFile['Date'] = pd.to_datetime(tempHoldingFile['Date'])
                    #start_date = tempHoldingFile['Date'].min()
                    end_date = pd.to_datetime(end_date)

                    tempHoldingFile.to_excel("tempHoldingFile.xlsx", index = False)
                    tempUnmatchedStocks.to_excel("tempUnmatchedStocks.xlsx", index = False)

                    
                    ## Fetching the price and CurrentSP
                    input_file = "tempHoldingFile.xlsx"  # Path to the input Excel file
                    stock_column = "SECURITY_ID"       # Column name for stock tickers
                    start_date_column = "Date" # Column name for start dates
                    # start_date_jan = "2023-01-09"
                    # start_date_jan_ = "2023-01-10"

                    #start_date_jan = start_date.date()
                    start_date_jan = start_date
                    start_date_jan_ = start_date_jan + timedelta(days=1)

                    end_date_present = "2024-12-31"
                    end_date_present_ = "2025-01-01"     # Column name for end dates
                    output_file = "tempHoldingFile.xlsx"  # Path for the output Excel file

                    fetch_prices(input_file, stock_column, start_date_column, start_date_jan, start_date_jan_, end_date_present, end_date_present_, output_file)










                df = pd.read_excel("tempHoldingFile.xlsx")

                
                #print(df)



                #df = pd.read_excel(uploaded_file, usecols=['SECURITY_ID', 'QUANTITY','Date', 'Company name', '16 Sectors', 'SEBI Mcap Class', 'PE', 'PB', 'Price', 'Div Yield(%)', 'DivPayOut', 'CurrentMC', 'CurrentSP', 'EPSM22', 'EPSJ22', 'EPSS22', 'EPSD22', 'Market Cap(cr)'])
                df.dropna(inplace=True)  # Remove rows with NaN values
                df['Date'] = pd.to_datetime(df['Date'])
                #start_date = df['Date'].min()


                ## Dynamic Start date
                #start_date = start_date.date()
                start_date = pd.to_datetime(start_date)
                start_date = start_date.date()
                start_date = max(start_date, fixed_start_date)

                end_date = end_date.date()
                if start_date > end_date:
                    st.error("Given start_date is greater than given end_date")

                # else:
                #     st.write("Start date and end date are within the timeframe")




                end_date = pd.to_datetime(end_date)


                print(df)







                stock_symbols = df['SECURITY_ID'].tolist()

                #st.write(stock_symbols)

                stock_names = []
                stock_name_n_symbol = {}

                

                for symbol in stock_symbols:
                    ticker = yf.Ticker(symbol)
                    stock_name = ticker.info['longName']
                    stock_names.append(stock_name)
                    stock_name_n_symbol[stock_name] = symbol
                df['yfName'] = stock_names
                #num_stocks = df['Quantity']
                #st.write(num_stocks.reset_index(drop=True))
                #st.write(num_stocks)
                historical_data = {}
                for symbol in stock_symbols:
                    #stock_data = yf.download(symbol, start=start_date, end=end_date)[['Close']]
                    historical_data[symbol] = yf.download(symbol, start=start_date, end=end_date, multi_level_index = False)['Close']
                    #historical_data[symbol] = stock_data['Close'].squeeze()
            
                adj_close_df = pd.DataFrame(historical_data)
                adj_close_df.to_csv('adj_close_df.csv')
                #st.write("Adjusted Closing Prices")
                
                #st.write(adj_close_df)

            if select_benchmark == 'NIFTY 50':
                excel_file_nifty50 = pd.read_excel("./benchmarks/nifty50Benchmark.xlsx")
                excel_file_nifty50.to_csv("./benchmarks/nifty50Benchmark.csv", index = False)
                csv_file = "./benchmarks/nifty50Benchmark.csv"
                benchdata = yf.download("^NSEI", start = start_date, end = end_date, multi_level_index = False)['Close']
                
                #benchdata = benchdata['Close']['^NSEI']
                sectorbenchmark = pd.read_csv('./benchmarks/nifty50SectorWeightages.csv')
                name = 'NIFTY 50'

            if select_benchmark == 'NIFTY 100':
                
                csv_file = "./benchmarks/nifty100Benchmark.csv"
                benchdata = yf.download("^CNX100", start = start_date, end = end_date, multi_level_index = False)['Close']
                sectorbenchmark = pd.read_csv('./benchmarks/nifty100SectorWeightages.csv')
                name = 'NIFTY 100'
            
            if select_benchmark == 'NIFTY 500':
                
                csv_file = "./benchmarks/nifty500Benchmark.csv"
                benchdata = yf.download("^CRSLDX", start = start_date, end = end_date, multi_level_index=False)['Close']
                sectorbenchmark = pd.read_csv('./benchmarks/nifty500SectorWeightages.csv')
                name = 'NIFTY 500'

            if select_benchmark == 'SENSEX 50':
                
                csv_file = "./benchmarks/sensex50Benchmark.csv"
                benchdata = yf.download("SNSX50.BO", start = start_date, end = end_date, multi_level_index=False)['Close']
                sectorbenchmark = pd.read_csv('./benchmarks/sensex50SectorWeightages.csv')
                name = 'SENSEX 50'

            if select_benchmark == 'BSE 100':
                
                csv_file = "./benchmarks/bse100Benchmark.csv"
                benchdata = yf.download("BSE-100.BO", start = start_date, end = end_date, multi_level_index=False)['Close']
                sectorbenchmark = pd.read_csv('./benchmarks/bse100SectorWeightages.csv')
                name = 'BSE 100'

            if select_benchmark == 'BSE 500':
                
                csv_file = "./benchmarks/bse500Benchmark.csv"
                benchdata = yf.download("BSE-500.BO", start = start_date, end = end_date, multi_level_index=False)['Close']
                sectorbenchmark = pd.read_csv('./benchmarks/bse500SectorWeightages.csv')
                name = 'BSE 500'

            

            
            
            dfForStartDate = pd.read_excel("tempHoldingFile.xlsx")
            dfForStartDate.dropna(inplace=True)  # Remove rows with NaN values
            dfForStartDate['Date'] = pd.to_datetime(dfForStartDate['Date'])
            #start_date = dfForStartDate['Date'].min()
            end_date = pd.to_datetime(end_date)

            # if select_benchmark == 'NIFTY 50':
            #     #csv_file = "NIFTY 50 Stock Weightages.csv"
            #     excel_file_nifty50 = pd.read_excel("./benchmarks/nifty50Benchmark.xlsx")
            #     excel_file_nifty50.to_csv("./benchmarks/nifty50Benchmark.csv", index = False)

            #     csv_file = "./benchmarks/nifty50Benchmark.csv"
            #     benchdata = yf.download("^NSEI", start = start_date, end = end_date)['Adj Close']
            #     sectorbenchmark = pd.read_csv('NIFTY 50 Sector Weightages.csv')
            #     name = 'NIFTY 50'
            # if select_benchmark == 'NSE 500':
            #     csv_file = "NSE 500 Stock Weightages.csv"
            #     benchdata = yf.download("^CRSLDX", start = start_date, end = end_date)['Adj Close']
            #     sectorbenchmark = pd.read_csv('NSE 500 Sector Weightages.csv')
            #     name = 'NSE 500'
            # if select_benchmark == 'HDFC NIFTY 50':
            #     csv_file = "HDFC NIFTY 50.csv"
            #     benchdata = yf.download("HDFCNIFETF.NS", start = start_date, end = end_date)['Adj Close']
            #     sectorbenchmark = pd.read_csv('NIFTY 50 Sector Weightages.csv')
            #     name = 'HDFC NIFTY 50'
            # elif select_benchmark == 'ICICI Prudential Nifty 100 ETF Fund':
            #     csv_file = "ICICIBenchmark.csv"
            #     #benchdata = yf.download("ICICINF100.NS", start = start_date, end = end_date)['Adj Close']
            #     benchdata = pd.read_csv('ICICI_Nifty100ETF.csv')
            #     benchdata['Date'] = pd.to_datetime(benchdata['Date'], format='%d-%b-%Y')
            #     benchdata = benchdata.sort_values(by='Date', ascending=True).reset_index(drop=True)
            #     benchdata = benchdata[(benchdata['Date'] >= start_date) & (benchdata['Date'] <= end_date)]
            #     sectorbenchmark = pd.read_csv('ICICIBenchmark.csv')
            #     name = 'ICICI Prudential Nifty 100 ETF Fund'
                
            
            #st.write(benchdata)

            class cfg:
                hpfilter_lamb = 6.25
                q = 1.0
                fmin = 0.001
                fmax = 0.5
                num_stocks = len(adj_close_df.columns)
            
            stock_prices_for_algo = adj_close_df.reset_index(drop=True)
            stock_prices_for_algo = stock_prices_for_algo.apply(lambda x: (x / x.iloc[0]) * 100, axis=0)
            normalized_df_long = stock_prices_for_algo.reset_index().melt(id_vars='index', var_name='Stock', value_name='Normalized Price')

            fig_1 = px.line(normalized_df_long, x='index', y='Normalized Price', color='Stock',
                labels={'index': 'Days', 'Normalized Price': 'Price (Normalized to 100)', 'Stock': 'Stock Symbol'},
                title='Normalized Stock Prices Over Time')
            
            fig_1.update_layout(
                width=1000,  # Set the width of the chart
                height=600,  # Set the height of the chart
                title_font_size=24,  # Increase the title font size
                legend_title_font_size=16,  # Increase the legend title font size
                legend_font_size=14,  # Increase the legend font size
            )

            st.plotly_chart(fig_1)

            #a = pd.read_csv(csv_file)['Symbol'].to_list()
            a = pd.read_csv(csv_file)['SECURITY_ID'].dropna().to_list()
            print(a)
            b = yf.download(a, start = start_date, end = end_date, multi_level_index=False)['Close']

            ## Filling empty cells with '0'
            #b.fillna(0, inplace = True)

            b.to_csv('BenchmarkStockData.csv') #renamed in modular code to adj_close_df_benchmark.csv



            for s in adj_close_df.columns:
                cycle, trend = hpfilter(adj_close_df[s], lamb=cfg.hpfilter_lamb)
                adj_close_df[s] = trend
            

            log_returns = np.log(stock_prices_for_algo) - np.log(stock_prices_for_algo.shift(1))
            null_indices = np.where((log_returns.isna().sum() > 1).to_numpy())[0]
            drop_stocks = stock_prices_for_algo.columns[null_indices]
            log_returns = log_returns.drop(columns=drop_stocks)
            log_returns = log_returns.dropna()
            tickers = log_returns.columns
            cfg.num_stocks = len(tickers)
            mu = log_returns.mean().to_numpy() * 252
            print("mu")
            print(mu)
            sigma = log_returns.cov().to_numpy() * 252

            cfg.kappa = cfg.num_stocks
            


            def objective_mvo_miqp(trial, _mu, _sigma):
                cpo = ClassicalPO(_mu, _sigma, cfg)
                cpo.cfg.gamma = trial.suggest_float('gamma', 0.0, 1.5)
                res = cpo.mvo_miqp()
                mvo_miqp_res = cpo.get_metrics(res['w'])
                del cpo
                return mvo_miqp_res['sharpe_ratio']
            
            study_mvo_miqp = optuna.create_study(
                study_name='classical_mvo_miqp',
                direction='maximize',
                pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=50),
                load_if_exists=True)
            
            study_mvo_miqp.optimize(lambda trial: objective_mvo_miqp(trial, mu, sigma), n_trials=25-len(study_mvo_miqp.trials), n_jobs=1)
            trial_mvo_miqp = study_mvo_miqp.best_trial
            cpo = ClassicalPO(mu, sigma, cfg)
            cpo.cfg.gamma = 1.9937858736079478
            res = cpo.mvo_miqp()
            weights = res['w']

            for s in b.columns:
                cycle, trend = hpfilter(b[s], lamb=cfg.hpfilter_lamb)
                b[s] = trend

            log_returns_1 = np.log(b) - np.log(b.shift(1))
            null_indices_1 = np.where((log_returns_1.isna().sum() > 1).to_numpy())[0]
            drop_stocks_1 = b.columns[null_indices_1]
            log_returns_1 = log_returns_1.drop(columns=drop_stocks_1)
            log_returns_1 = log_returns_1.dropna()
            tickers_1 = b.columns

            cfg.num_stocks_1 = len(tickers_1)
            st.write("cfg")
            st.write(cfg.num_stocks)
            st.write(cfg.num_stocks_1)
            mu_1 = log_returns_1.mean().to_numpy() * 252
            sigma_1 = log_returns_1.cov().to_numpy() * 252
            cfg.kappa = cfg.num_stocks_1

            cpo_1 = ClassicalPO(mu_1, sigma_1, cfg)
            c = pd.read_csv(csv_file)
            weights_1 = c['Weightage(%)'].to_list()
            weights_1 = [weight / 100 for weight in weights_1]
            #st.write(weights_1)

            mvo_miqp_bench = cpo_1.get_metrics(weights_1)
            #st.write(mvo_miqp_bench)


            first_row_adj_close = adj_close_df.iloc[0]
            total_budget = (first_row_adj_close * df['QUANTITY'].values).sum()
            #st.write(total_budget)
            wt_stock = ((first_row_adj_close * df['QUANTITY'].values) / total_budget) 
            #st.write(df)
            #df['Weightage(%)'] = ((first_row_adj_close * df['QUANTITY'].values) / total_budget) * 100



            #portfolio_dict = dict(zip(df['yfName'], wt_stock))
            portfolio_dict = dict(zip(df["Company name"], wt_stock))




            #st.write(portfolio_dict)
            df['Weightage(%)'] = portfolio_dict.values()
            #st.write(df)
            vp_list = list(portfolio_dict.values())
            portfolio_array = np.array(vp_list)
            #st.write(vp_list)

            rows = len(stock_symbols)
            wt_bench = pd.read_csv(csv_file).reset_index(drop=True) #nrows should be dynamic
            #st.write(wt_bench)
            wt_bench_filtered = wt_bench[['Company name', 'Weightage(%)']]
            #st.write(wt_bench_filtered)
            benchmark_dict = dict(zip(wt_bench_filtered.iloc[:, 0], wt_bench_filtered.iloc[:, 1]))
            #st.write(benchmark_dict)
            #st.write(benchmark_dict)
            #st.write(len(benchmark_dict.keys()))
            bp_list = list(benchmark_dict.values())
            #st.write(bp_list)
            #st.write(len(bp_list))
            benchmark_array = np.array(bp_list)

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Avg % Wgt Portfolio:**")
                # for stock, value in portfolio_dict.items():
                #     percentage = round(value*100, 2)  # Multiply by 100 and round off to 2 decimal places
                #     st.text(f"{stock:<45}{percentage:>15}")
                portfolio_df = pd.DataFrame({
                    'Stock': list(portfolio_dict.keys()),
                    'Weight (%)': [round(value * 100, 2) for value in portfolio_dict.values()]})
                #portfolio_weights = portfolio_df['Weight (%)'].to_list()
                st.write(portfolio_df.set_index('Stock'))
    
                st.markdown("**Returns and Risk of Portfolio:**")
                mvo_miqp_bench = cpo.get_metrics(portfolio_array)
                # for metric, value in mvo_miqp_bench.items():
                #     if metric in ['returns', 'risk']:
                #         display_value = round(value*100 ,2)
                #     else:
                #         display_value = round(value, 2)
                #     st.text(f"{metric:<45}{display_value:>15}")
                metrics_df = pd.DataFrame({
                    'Metric': list(mvo_miqp_bench.keys()),
                    'Value': [round(value * 100, 2) if metric in ['returns', 'risk'] else round(value, 2) 
                        for metric, value in mvo_miqp_bench.items()]})
                st.write(metrics_df.set_index('Metric'))


            with col2:
                st.markdown("**Avg % Wgt Benchmark:**")
                # for stock, value in benchmark_dict.items():
                #     percentage = round(value, 2)
                #     st.text(f"{stock:<35}{percentage:>15}")
                portfolio_df_1 = pd.DataFrame({
                    'Stock': list(benchmark_dict.keys()),
                    'Weight (%)': [round(value, 2) for value in benchmark_dict.values()]})
                #benchmark_weights = portfolio_df_1['Weight (%)'].tolist()
                st.write(portfolio_df_1.set_index('Stock'))
                st.markdown("**Returns and Risk of Benchmark:**")
                #st.write(benchmark_array)
                #st.write(len(benchmark_array))
                mvo_miqp_bench_1 = cpo_1.get_metrics(benchmark_array)
                # for metric, value in mvo_miqp_bench.items():
                #     if metric in ['returns', 'risk']:
                #         display_value = round(value ,2)
                #     else:
                #         display_value = round(value, 2)
                #     st.text(f"{metric:<35}{display_value:>15}")
                metrics_df_1 = pd.DataFrame({
                    'Metric': list(mvo_miqp_bench_1.keys()),
                    'Value': [round(value, 2) if metric in ['returns', 'risk'] else round(value, 2) 
                        for metric, value in mvo_miqp_bench_1.items()]})
                st.write(metrics_df_1.set_index('Metric'))

            colors = ['gold', 'mediumturquoise', 'darkorange', 'lightgreen', 'DarkOrchid', 'DeepPink', 'Maroon', 'MistyRose', 'Olive', 'Salmon' ]
            fig = go.Figure(data=[go.Pie(labels=list(portfolio_dict.keys()), values=list(portfolio_dict.values()), hole=.3)])
            fig.update_traces(hoverinfo='label+percent',textfont_size=15, marker=dict(colors=colors, line=dict(color='#000000', width=2)))
            st.markdown("**Pie Chart of Stock Weights Allocated Portfolio:**")
            st.plotly_chart(fig)


            ############################################# Added code for eliminating the dependency on stocks_by_sector_json

            #def getSectorWeightages():

            ############################################### Sector weights portfolio
            sectoral_end_date_sum = {}
            sectoral_start_date_sum = {}
            sector_weights_axis= {}

            for index, row in df.iterrows():
                sector = row['sector']

                # For adding the sectoral prices on last day and first day. To use in returns of portfolio sector
                if sector not in sectoral_end_date_sum.keys():
                    sectoral_end_date_sum[sector] = adj_close_df[row['SECURITY_ID']].iloc[-1]
                else:
                    sectoral_end_date_sum[sector] += adj_close_df[row['SECURITY_ID']].iloc[-1]
                
                if sector not in sectoral_start_date_sum.keys():
                    sectoral_start_date_sum[sector] = adj_close_df[row['SECURITY_ID']].iloc[0]
                else:
                    sectoral_start_date_sum[sector] += adj_close_df[row['SECURITY_ID']].iloc[0]

                
                if sector not in sector_weights_axis.keys():
                    sector_weights_axis[sector] = row['Weightage(%)']
                
                else:
                    sector_weights_axis[sector] += row['Weightage(%)']
            
            returns_of_portfolio_sector = {}

            for sector in sectoral_end_date_sum.keys(): 
                if sectoral_end_date_sum[sector] != 0:
                    returns_of_portfolio_sector[sector] = ((sectoral_end_date_sum[sector] - sectoral_start_date_sum[sector])/sectoral_start_date_sum[sector]) * 100


            keys_axis = sector_weights_axis.keys()
            values_sector_axis = sector_weights_axis.values()
            #portfolio_weights = list(keys_axis)
            #print(sector_weights_axis)
            fig_sector_axis = go.Figure(data=[go.Pie(labels=list(keys_axis),values=list(values_sector_axis), hole=.3)])
            fig_sector_axis.update_traces(hoverinfo='label+percent',textfont_size=15, marker=dict(line=dict(color='#000000', width=2)))
            st.markdown("**Pie Chart of Sector Weights Portfolio:**")
            st.plotly_chart(fig_sector_axis)

            ###################################### end of sector weights portfolio

            #################################### Beginning on sector weights Benchmark

            fig_bench = go.Figure(data=[go.Pie(labels=list(benchmark_dict.keys()), values=list(benchmark_dict.values()), hole=.3)])
            fig_bench.update_traces(hoverinfo='label+percent',textfont_size=15, marker=dict(colors=colors, line=dict(color='#000000', width=2)))
            st.markdown("**Pie Chart of Stock Weights Allocated Benchmark:**")
            st.plotly_chart(fig_bench)


            dfForGettingSectorWeightsBenchmark = pd.read_csv(csv_file)

            #sector_weights = {}

            dfForGettingSectorWeightsBenchmark = dfForGettingSectorWeightsBenchmark.dropna(subset=["sector", "Weightage(%)"])

            sector_weights_df = dfForGettingSectorWeightsBenchmark.groupby("sector")["Weightage(%)"].sum()

            sector_weights = sector_weights_df.to_dict()

            


                



            





























            ##################################################### Previous Sectors code

            # with open('stocks_by_sector.json', 'r') as file:
            #     data = json.load(file)
            
            # sector_weights_axis= {}
            # sectors = data
            # #st.write(sectors)
            # sectoral_end_date_sum = {}  # Added code
            # sectoral_start_date_sum = {} # Added code
            # for keys in sectors.keys():
            #     sectoral_end_date_sum[keys] = 0
            #     sectoral_start_date_sum[keys] = 0
            # #st.write(sectoral_end_date_sum)
            # #st.write(portfolio_dict)

            # for stock, weight in portfolio_dict.items():
            #     for sector, stocks_in_sector in sectors.items():
            #         if stock in stocks_in_sector:
            #             sector_weights_axis.setdefault(sector, 0)
            #             sector_weights_axis[sector] += weight
            #             #st.write(sector_weights_axis)
            #             sym = stock_name_n_symbol[stock]
            #             #st.write(sym)
            #             sectoral_end_date_sum[sector] += adj_close_df[sym].iloc[-1]
            #             sectoral_start_date_sum[sector] += adj_close_df[sym].iloc[0]

            # #st.write(sectoral_end_date_sum)
            # returns_of_portfolio_sector = {} # Added code

            # for sector in sectoral_end_date_sum.keys(): # Added code
            #     if sectoral_end_date_sum[sector] != 0:
            #         returns_of_portfolio_sector[sector] = ((sectoral_end_date_sum[sector] - sectoral_start_date_sum[sector])/sectoral_start_date_sum[sector]) * 100


            # keys_axis = sector_weights_axis.keys()
            # values_sector_axis = sector_weights_axis.values()
            # #portfolio_weights = list(keys_axis)
            # #print(sector_weights_axis)
            # fig_sector_axis = go.Figure(data=[go.Pie(labels=list(keys_axis),values=list(values_sector_axis), hole=.3)])
            # fig_sector_axis.update_traces(hoverinfo='label+percent',textfont_size=15, marker=dict(line=dict(color='#000000', width=2)))
            # st.markdown("**Pie Chart of Sector Weights Portfolio:**")
            # st.plotly_chart(fig_sector_axis)

            ###################################################################### Previous Benchmark sector

            # fig_bench = go.Figure(data=[go.Pie(labels=list(benchmark_dict.keys()), values=list(benchmark_dict.values()), hole=.3)])
            # fig_bench.update_traces(hoverinfo='label+percent',textfont_size=15, marker=dict(colors=colors, line=dict(color='#000000', width=2)))
            # st.markdown("**Pie Chart of Stock Weights Allocated Benchmark:**")
            # st.plotly_chart(fig_bench)

            # sector_weights= {}
            # sectors_bench = data
            # for stock, weight in benchmark_dict.items():
            #     for sector, stocks_in_sector in sectors_bench.items():
            #         if stock in stocks_in_sector:
            #             #st.write(stock)
            #             sector_weights.setdefault(sector, 0)
            #             #st.write(sector)
            #             sector_weights[sector] += weight
            #st.write(sector_weights)
            
            








            
            keys = sector_weights.keys()
            values_sector = sector_weights.values()
            #benchmark_weights = list(keys)
            fig_sector = go.Figure(data=[go.Pie(labels=list(keys),values=list(values_sector), hole=.3)])
            fig_sector.update_traces(hoverinfo='label+percent',textfont_size=15, marker=dict(line=dict(color='#000000', width=2)))
            st.markdown("**Pie Chart of Sector Weights Benchmark:**")
            st.plotly_chart(fig_sector)

            ## Code for bar graph
            def compare_return_risk(metrics_1, metrics_2, title):
                metric_labels = ["returns", "risk"]

                # Values from metrics_1
                values_1 = [round(metrics_1['returns'], 2), round(metrics_1['risk'], 2)]

                # Values from metrics_2
                values_2 = [round(metrics_2['returns']*100,2), round(metrics_2['risk']*100,2)]

                # Create Bar Traces
                trace1 = go.Bar(x=metric_labels, y=values_1, name="Benchmark", text = values_1, textposition='auto', marker_color="green")
                trace2 = go.Bar(x=metric_labels, y=values_2, name="Portfolio", text = values_2, textposition='auto', marker_color="red")

                # Create Layout
                layout = go.Layout(
                    title=title,
                    xaxis=dict(title="Metrics"),
                    yaxis=dict(title="Values in %"),
                    barmode="group"  # Side-by-side bars
                )

                # Create Figure
                fig = go.Figure(data=[trace1, trace2], layout=layout)

                st.plotly_chart(fig)


                return
            
            def compare_Sharpe_ratio_and_diversification_ratio(metrics_1, metrics_2, title):
                metric_labels = ["sharpe_ratio", "diversification_ratio"]

                # Values from metrics_1
                values_1 = [round(metrics_1['sharpe_ratio'],2), round(metrics_1['diversification_ratio'],2)]

                # Values from metrics_2
                values_2 = [round(metrics_2['sharpe_ratio'],2), round(metrics_2['diversification_ratio'],2)]

                # Create Bar Traces
                trace1 = go.Bar(x=metric_labels, y=values_1, name="Benchmark", text = values_1, textposition='auto', marker_color="green")
                trace2 = go.Bar(x=metric_labels, y=values_2, name="Portfolio", text = values_2, textposition='auto', marker_color="red")

                # Create Layout
                layout = go.Layout(
                    title=title,
                    xaxis=dict(title="Metrics"),
                    yaxis=dict(title="Values"),
                    barmode="group" , # Side-by-side bars
                )

                # Create Figure
                fig = go.Figure(data=[trace1, trace2], layout=layout)

                st.plotly_chart(fig)


                return
            
            # Comparison of return-risk 
            compare_return_risk(mvo_miqp_bench_1, mvo_miqp_bench, "Return, Risk : Benchmark vs Portfolio")
            # comparison of sharpe ratio and diversification ratio
            compare_Sharpe_ratio_and_diversification_ratio(mvo_miqp_bench_1, mvo_miqp_bench, "Sharpe ratio, Diversification ratio : Benchmark vs Portfolio")



            st.markdown('<p style="font-size:20px;"><b>Line chart of Portfolio against the Benchmark (Rebased to 100 for initial date)</b></p>', unsafe_allow_html=True)
            quantity_dict = pd.Series(df.QUANTITY.values, index=df.SECURITY_ID).to_dict()
            for symbol in adj_close_df.columns[1:]:  # Skip the 'Date' column (index 0)
                if symbol in quantity_dict:
                    adj_close_df[symbol] = adj_close_df[symbol] #* quantity_dict[symbol]
            adj_close_df['Portfolio Value'] = adj_close_df.iloc[:, 1:].sum(axis=1)

            
      ##      ########### Original code of the portfolio return calculation - 
            adj_close_df['Return'] = (adj_close_df['Portfolio Value'] / adj_close_df['Portfolio Value'][0]) * 100
            #st.write(adj_close_df['Return'])

     ##     Added code
            #adj_close_df['Return'] = np.log(adj_close_df['Portfolio Value'] / adj_close_df['Portfolio Value'].shift(1))

            #adj_close_df['Return'] = adj_close_df['Portfolio Value']
      ##      
            benchdata.to_csv("TotalBenchmarkValue.csv")
            benchdata = pd.read_csv("TotalBenchmarkValue.csv")
            benchdata = benchdata.set_index(('Date'))
            
      ##    Original code of Benchmark return calculation      
            benchdata['Return'] = (benchdata['Close']/benchdata['Close'].iloc[0]) * 100
            #st.write(benchdata['Return'])

     ##     Added code
            #benchdata['Return'] = np.log(benchdata['Adj Close']/ benchdata['Adj Close'].shift(1))
            #benchdata['Return'] = benchdata['Adj Close']

     ##      

            fig_compare = go.Figure()
            fig_compare.add_trace(go.Scatter(x= adj_close_df.index, 
                    y=  adj_close_df['Return'],
                    mode='lines+markers', 
                    name='Return Portfolio', 
                    #name = 'Return Benchmark',
                    line=dict(color='red')))
            
            fig_compare.add_trace(go.Scatter(x=benchdata.index, 
                    y=benchdata['Return'], 
                    mode='lines+markers', 
                    name='Return Benchmark',
                    #name = 'Return Portfolio', 
                    line=dict(color='blue')))
            
            fig_compare.update_layout(title='Return Over Time',
                    xaxis_title='Date', 
                    yaxis_title='Return',
                    autosize=False, 
                    width=1000, 
                    height=600,
                    #yaxis=dict(range=[115, 145]
                    )
            st.plotly_chart(fig_compare)

            st.markdown('<p style="font-size:20px;"><b>Portfolio Weight vs Benchmark Weight</b></p>', unsafe_allow_html=True)
            all_sectors = set(sector_weights.keys()).union(set(sector_weights_axis.keys()))
            #st.write(all_sectors)
            # Create a DataFrame
            data = {
            "Sector": [],
            "Portfolio Weight": [],
            "Benchmark Weight": [],
            "Status": [],
            "+/-": []
            }
            sector_weights_axis = {key: value * 100 for key, value in sector_weights_axis.items()}
            #st.write(sector_weights_axis)
            #st.write(sector_weights)
            for sector in all_sectors:
                port_weight = sector_weights_axis.get(sector, 0)
                bench_weight = sector_weights.get(sector, 0)
                diff = port_weight - bench_weight
            # Determine the status (Overweight, Underweight)
                if diff > 0:
                    status = "Overweight"
                elif diff < 0:
                    status = "Underweight"
                else:
                    status = "Equal"
    
            # Add data to the dictionary for each sector
                data["Sector"].append(sector)
                data["Portfolio Weight"].append(round(port_weight, 2))
                data["Benchmark Weight"].append(round(bench_weight, 2))
                data["Status"].append(status)
                data["+/-"].append(round(diff, 2))

            # Convert to DataFrame
            df_result = pd.DataFrame(data)

            # Display the final DataFrame
            df_result = df_result[['Sector', 'Portfolio Weight', 'Benchmark Weight', 'Status', '+/-']]
            df_result.sort_values(by='Sector', inplace=True)
            st.dataframe(df_result.set_index('Sector'))

            st.markdown('<p style="font-size:20px;"><b>Sebi Classification-wise Weights</b></p>', unsafe_allow_html=True)
            cap_class_counts = df['SEBI Mcap Class'].value_counts()
            labels_port = cap_class_counts.index.tolist()  # ['Large Cap', 'Mid Cap', 'Small Cap']
            sizes_port = cap_class_counts.values.tolist()  # The count of each class
            colors = ['gold', 'DeepPink', 'SkyBlue']  # You can adjust the colors here
            fig_sector_port = go.Figure(data=[go.Pie(labels=labels_port, values=sizes_port, hole=.3)])

            fig_sector_port.update_traces(
                hoverinfo='label+percent', 
                textfont_size=15, 
                marker=dict(colors=colors, line=dict(color='#000000', width=2)))
            
            # Display the pie chart with title in Streamlit
            st.markdown("**Portfolio Weight Distribution**")
            st.plotly_chart(fig_sector_port)

            st.markdown('<p style="font-size:20px;"><b>Top 10 gainers and Bottom 10 laggers (Based on Return)</b></p>', unsafe_allow_html=True)
            top10gainersandlaggers = pd.read_csv('adj_close_df.csv')
            top10_withoutdate= top10gainersandlaggers.drop(columns=['Date'])

            for s in top10_withoutdate.columns:
                cycle, trend = hpfilter(top10_withoutdate[s], lamb=cfg.hpfilter_lamb)
                top10_withoutdate[s] = trend

            log_returns = np.log(top10_withoutdate) - np.log(top10_withoutdate.shift(1))
            null_indices = np.where((log_returns.isna().sum() > 1).to_numpy())[0]
            drop_stocks = top10_withoutdate.columns[null_indices]
            log_returns = log_returns.drop(columns=drop_stocks)
            log_returns = log_returns.dropna()
            tickers = log_returns.columns 
            cfg.num_stocks = len(tickers)
            mu = log_returns.mean()* len(top10gainersandlaggers['Date'])
            sigma = log_returns.cov().to_numpy() * len(top10gainersandlaggers['Date'])

            col3, col4 = st.columns(2)
            top_10_gainers = pd.DataFrame({
                    'Stock': mu.sort_values(ascending=False).head(10).index,
                    'Return(%)': mu.sort_values(ascending=False).head(10).apply(lambda x: f"{x:.2f}%")})
            
            # Create a dataframe for the bottom 10 laggers
            bottom_10_laggers = pd.DataFrame({
            'Stock': mu.sort_values().head(10).index,
            'Return(%)': mu.sort_values().head(10).apply(lambda x: f"{x:.2f}%")})

            with col3:
                st.markdown('**Top 10 gainers**')
                st.dataframe(top_10_gainers, use_container_width=True, hide_index=True)
            with col4:
                st.markdown('**Bottom 10 laggers**')
                st.dataframe(bottom_10_laggers, use_container_width=True, hide_index=True)

            st.markdown('<p style="font-size:20px;"><b>Top 10 holdings, Bottom 10 holdings (With performance of 1m, 3m, 6m, 1 yr)</b></p>', unsafe_allow_html=True)
            top10gainersandlaggers['Date'] = pd.to_datetime(top10gainersandlaggers['Date'])
            def calculate_returns(dataframe, days):
                log_returns = np.log(dataframe) - np.log(dataframe.shift(1))
                null_indices = np.where((log_returns.isna().sum() > 1).to_numpy())[0]
                drop_stocks = dataframe.columns[null_indices]
                log_returns = log_returns.drop(columns=drop_stocks)
                log_returns = log_returns.dropna()
                tickers = log_returns.columns
                mu = log_returns.mean() * days
                return mu
            start_date_1m = pd.to_datetime(start_date)
            end_date_1m = start_date_1m + pd.DateOffset(months=1)
            filtered_df_1m = top10gainersandlaggers[(top10gainersandlaggers['Date'] >= start_date_1m) & (top10gainersandlaggers['Date'] <= end_date_1m)].drop(columns=['Date'])

            start_date_3m = pd.to_datetime(start_date)
            end_date_3m = start_date_3m + pd.DateOffset(months=3)
            filtered_df_3m = top10gainersandlaggers[(top10gainersandlaggers['Date'] >= start_date_3m) & (top10gainersandlaggers['Date'] <= end_date_3m)].drop(columns=['Date'])

            start_date_6m = pd.to_datetime(start_date)
            end_date_6m = start_date_6m + pd.DateOffset(months=6)
            filtered_df_6m = top10gainersandlaggers[(top10gainersandlaggers['Date'] >= start_date_3m) & (top10gainersandlaggers['Date'] <= end_date_3m)].drop(columns=['Date'])

            start_date_1y = pd.to_datetime(start_date)
            end_date_1y = start_date_1y + pd.DateOffset(months=12)
            filtered_df_1y = top10gainersandlaggers[(top10gainersandlaggers['Date'] >= start_date_3m) & (top10gainersandlaggers['Date'] <= end_date_3m)].drop(columns=['Date'])

            start_date_port = pd.to_datetime(start_date)
            end_date_port = pd.to_datetime(end_date)
            filtered_df = top10gainersandlaggers[(top10gainersandlaggers['Date'] >= start_date_port) & (top10gainersandlaggers['Date'] <= end_date_port)].drop(columns=['Date'])
            working_days = np.busday_count(start_date_port.date(), end_date_port.date())

            returns_1m = calculate_returns(filtered_df_1m, 21)
            returns_3m = calculate_returns(filtered_df_3m, 63)
            returns_6m = calculate_returns(filtered_df_6m, 126)
            returns_1y = calculate_returns(filtered_df_1y, 252)
            returns_total = calculate_returns(filtered_df, working_days)

            weights = portfolio_dict
            stock_ticker_map = dict(zip(portfolio_dict.keys(), stock_symbols))
            top_stocks = list(weights.keys())
            data = {
            'Company': list(weights.keys()),
            'Weightages(%)': list(weights.values()),
            '1 Month Return(%)': [returns_1m.get(stock_ticker_map[stock], np.nan)*100 for stock in weights.keys()],
            '3 Month Return(%)': [returns_3m.get(stock_ticker_map[stock], np.nan)*100 for stock in weights.keys()],
            '6 Month Return(%)': [returns_6m.get(stock_ticker_map[stock], np.nan)*100 for stock in weights.keys()],
            '1 Year Return(%)': [returns_1y.get(stock_ticker_map[stock], np.nan)*100 for stock in weights.keys()],
            'Total Return(%)': [returns_total.get(stock_ticker_map[stock], np.nan)*100 for stock in weights.keys()]
            }
        
            combined_data = pd.DataFrame(data)
            combined_data = combined_data.sort_values(by='Weightages(%)', ascending=False)
            st.write(combined_data.set_index('Company'))

            st.markdown('<p style="font-size:20px;"><b>Top 10 contribution to returns, Bottom 10 contribution to returns</b></p>', unsafe_allow_html=True)
            port_weights = portfolio_dict

            total_returns = dict(zip(combined_data['Company'], combined_data['Total Return(%)'])) 

            contribution_return_port = {
            key: (port_weights[key] * total_returns[key])/100 for key in port_weights 
            }
            sorted_contribution_port = sorted(contribution_return_port.items(), key=lambda x: x[1])
            #st.write(contribution_return_port)
            df_port = pd.DataFrame(sorted_contribution_port, columns=['Company', 'Contribution to return(%)'])
            df_port = df_port.sort_values(by='Contribution to return(%)', ascending=False)
            df_port['Contribution to return(%)'] = df_port['Contribution to return(%)'] * 100



            #### 

            cr_data = pd.read_csv(csv_file)
            cr_tickers = cr_data['SECURITY_ID'].tolist()
            cr_weightages = cr_data['Weightage(%)'].tolist()
            cr_dict = dict(zip(cr_tickers, cr_weightages))
            #st.write(cr_dict)
            working_days = np.busday_count(start_date_port.date(), end_date_port.date())

            def calculate_top_contributors(tickers_weights, start_date, end_date):
                # Download stock data for the given date range
                stock_data = yf.download(list(tickers_weights.keys()), start=start_date, end=end_date, multi_level_index=False)['Close']
    
                # Calculate daily log returns for each stock
                log_returns = np.log(stock_data) - np.log(stock_data.shift(1))
                #print(log_returns['HDFCBANK.NS'])
    
                # Initialize a dictionary to store weighted contributions
                contributions = {}

                # Calculate weighted log returns
                for ticker, weight in tickers_weights.items():
                    if ticker in log_returns:
                        # Get the average log return for the stock
                        #log_returns = np.log(stock_data / stock_data.shift(1))
                        #null_indices = np.where((log_returns.isna().sum() > 1).to_numpy())[0]
                        #drop_stocks = stock_data.columns[null_indices]
                        #log_returns = log_returns.drop(columns=drop_stocks)
                        #log_returns = log_returns.dropna()
                        avg_log_return = log_returns[ticker].mean() * working_days
                        #print(avg_log_return)
                        #print(ticker)
                        #Calculate weighted contribution
                        contributions[ticker] = avg_log_return * weight

                contributions_df = pd.DataFrame(list(contributions.items()), columns=['Company', 'Contribution to return(%)'])

                # Sort by contribution and get the top contributors
                top_contributors_df = contributions_df.sort_values(by='Contribution to return(%)', ascending=False)

                return top_contributors_df
            
            top_contributors = calculate_top_contributors(cr_dict, start_date, end_date)
            #st.write(top_contributors)
            col5, col6 = st.columns(2)
            with col5:
                st.markdown('**Contribution to return - Portfolio, Top 10/Bottom 10**') 
                st.write(df_port.set_index('Company'))
            with col6:
                st.markdown('**Contribution to return - Benchmark, Top 10/Bottom 10**') 
                st.write(top_contributors.set_index('Company'))

            st.markdown('<p style="font-size:20px;"><b>Allocation Effect</b></p>', unsafe_allow_html=True)
            #st.latex('''Allocation: (w_i-W_i)(B_i-B)''')
            #st.latex('w_i:Portfolio weight of the sector')
            #st.latex('W_i:Benchmark weight of the sector')
            #st.latex('B_i:Benchmark return of the sector')
            #st.latex('B:Total Benchmark return')
            #sector_summary = sector_groups["Weightage(%)"].sum()  for calculating the sector weightages
            allocation_data = pd.read_csv(csv_file)
            #st.write(allocation_data)
            sector_groups = allocation_data.groupby("sector")

            benchmark_sector_dict = {}
            for sector, group in sector_groups:
                benchmark_sector_dict[sector] = group["Weightage(%)"].sum()

            price_data = pd.read_csv('BenchmarkStockData.csv')
            last_day_returns = price_data.iloc[-1]
            first_day_returns = price_data.iloc[0]

            allocation_data['last_day_returns'] = allocation_data['SECURITY_ID'].map(last_day_returns)
            allocation_data['first_day_returns'] = allocation_data['SECURITY_ID'].map(first_day_returns)

            sector_returns = allocation_data.groupby('sector').apply(
                lambda x: (((x['last_day_returns'].sum() - x['first_day_returns'].sum()) / x['first_day_returns'].sum() )*100 ) #Comment : #* #x['Weightage(%)'].sum()
                ).reset_index(name='sector_return')

            sector_returns_dict = sector_returns.set_index('sector')['sector_return'].to_dict()
            #st.write("Benchmark return of the sector")
            #st.write(sector_returns_dict)
            B = (benchdata.iloc[-1] - benchdata.iloc[0])/(benchdata.iloc[0]) * 100
            #st.write(B['Return'])
            #st.write("sector_weights_axis")
            #st.write(sector_weights_axis)
            #st.write("benchmark_sector_dict")
            #st.write(benchmark_sector_dict)
            #st.write("sector_returns_dict")
            #st.write(sector_returns_dict)

            allocation_effect = {}
            #st.write(sector_weights_axis)
            for sector in sector_weights_axis.keys():
                allocation_effect[sector] = (sector_weights_axis[sector] - benchmark_sector_dict[sector]) * (sector_returns_dict[sector] - B['Return']) #actual formula
                #allocation_effect[sector] = (sector_weights_axis[sector] - benchmark_sector_dict[sector]) * (sector_returns_dict[sector])
                allocation_effect[sector] = allocation_effect[sector]/100

            df_allocation_effect = pd.DataFrame(list(allocation_effect.items()), columns=['Sector', 'Allocation Effect(%)'])
            st.write(df_allocation_effect.set_index('Sector'))


            st.markdown('<p style="font-size:20px;"><b>Selection Effect</b></p>', unsafe_allow_html=True)
            selection_effect = {} # Added code
            #st.write("Portoflio return of the sector")
            #st.write(returns_of_portfolio_sector)
            for sector in returns_of_portfolio_sector.keys(): # Added code
                selection_effect[sector] = sector_weights_axis[sector] * (returns_of_portfolio_sector[sector] - sector_returns_dict[sector])
                selection_effect[sector] = selection_effect[sector] / 100
            
            df_selection_effect = pd.DataFrame(list(selection_effect.items()), columns=['Sector', 'Selection Effect(%)'])
            st.write(df_selection_effect.set_index('Sector'))


            st.markdown('<p style="font-size:20px;"><b>Beta</b></p>', unsafe_allow_html=True)
            end_date = end_date
            start_date_beta = datetime(end_date.year - 2, end_date.month, end_date.day)
            historical_data_beta = {}
            for symbol in stock_symbols:
                stock_data_beta = yf.download(symbol, start=start_date_beta, end=end_date, multi_level_index=False)['Close']
                historical_data_beta[symbol] = stock_data_beta

            adj_close_df_beta = pd.DataFrame(historical_data_beta)
            adj_close_df_beta.to_csv('adj_close_df_beta.csv')

            #stock_wt = dict(zip(df['SECURITY_ID'], wt_stock))
            stock_quantities = dict(zip(df['SECURITY_ID'],df['QUANTITY']))

            portfolio_values_port = adj_close_df_beta.apply(lambda row: sum(
            row[stock] * stock_quantities[stock] for stock in stock_quantities), axis=1)
            adj_close_df_beta['PV_Port'] = portfolio_values_port

            #st.write(adj_close_df_beta)

            historical_data_beta_bench = {}
            for symbol in cr_dict:
                stock_data_beta_bench = yf.download(symbol, start=start_date_beta, end=end_date, multi_level_index=False)['Close']
                historical_data_beta_bench[symbol] = stock_data_beta_bench
            adj_close_df_beta_bench = pd.DataFrame(historical_data_beta_bench)
            #adj_close_df_beta_bench = adj_close_df_beta_bench.dropna(axis=1, how='any') #removes nan columns
            adj_close_df_beta_bench = adj_close_df_beta_bench.fillna(0)

            #adj_close_df_beta_bench.index = pd.to_datetime(adj_close_df_beta_bench.index)
            #adj_close_df_beta_bench['Date'] = pd.to_datetime(adj_close_df_beta_bench['Date'], format='%Y-%m-%d')
            adj_close_df_beta_bench.to_csv('adj_close_df_beta_bench.csv')
            #st.write(adj_close_df_beta_bench)
            total_investment_beta = adj_close_df_beta_bench.iloc[0].sum()
            #st.write(total_investment_beta)
            investment_per_stock_beta_bench = {stock: total_investment_beta* weight/100 for stock, weight in cr_dict.items()}
            #st.write(investment_per_stock_beta_bench)
            #optimal_stocks_beta = {stock: investment // adj_close_df_beta_bench.iloc[0][stock] for stock, investment in investment_per_stock_beta_bench.items()}
            optimal_stocks_beta = {
                stock: (investment // adj_close_df_beta_bench.iloc[0][stock]) if adj_close_df_beta_bench.iloc[0][stock] != 0 else 0
                for stock, investment in investment_per_stock_beta_bench.items()}
            #st.write(optimal_stocks_beta)
            portfolio_value_beta = adj_close_df_beta_bench.apply(lambda row: sum(row[stock] * optimal_stocks_beta[stock] for stock in optimal_stocks_beta), axis=1)
            adj_close_df_beta_bench['PV_Bench'] = portfolio_value_beta
            #st.write(adj_close_df_beta_bench)


            slope, intercept, r_value, p_value, std_err = linregress(portfolio_value_beta, portfolio_values_port)
            beta = slope
            st.write(beta)

            st.markdown('<p style="font-size:20px;"><b>Standard Deviation</b></p>', unsafe_allow_html=True)
            adj_close_df_beta['PV_Port'] = (adj_close_df_beta['PV_Port'] - adj_close_df_beta['PV_Port'].shift(1)) / adj_close_df_beta['PV_Port'].shift(1)
            adj_close_df_beta_bench['PV_Bench'] = (adj_close_df_beta_bench['PV_Bench'] - adj_close_df_beta_bench['PV_Bench'].shift(1)) / adj_close_df_beta_bench['PV_Bench'].shift(1)
            adj_close_df_beta['PV_Port'] = adj_close_df_beta['PV_Port'].fillna(0)
            adj_close_df_beta_bench['PV_Bench'] = adj_close_df_beta_bench['PV_Bench'].fillna(0)
            std_port = adj_close_df_beta['PV_Port'].std() * np.sqrt(504) * 100
            std_bench = adj_close_df_beta_bench['PV_Bench'].std() * np.sqrt(504) * 100
            st.write('Standard Deviation of Portfolio:', std_port)
            st.write('Standard Deviation of Benchmark:', std_bench)

            st.markdown('<p style="font-size:20px;"><b>Max Draw Down</b></p>', unsafe_allow_html=True)
            def calculate_max_drawdown(series):
                roll_max = series.cummax()
                daily_drawdown = series / roll_max - 1.0
                max_drawdown = daily_drawdown.cummin()
                return max_drawdown.min()
            
            max_drawdown_port= calculate_max_drawdown(adj_close_df['Return'])
            max_drawdown_bench = calculate_max_drawdown(benchdata['Return'])

            st.write("Max Drawdown for Portfolio:", max_drawdown_port*100)
            st.write("Max Drawdown for Benchmark:", max_drawdown_bench*100)

            st.markdown('<p style="font-size:20px;"><b>Sortino Ratio</b></p>', unsafe_allow_html=True)
            adj_close_df['Shift'] = (adj_close_df['Return'] - adj_close_df['Return'].shift(1))
            adj_close_df['Shift'] = adj_close_df['Shift'].fillna(0)
            adj_close_df['Shift'] = adj_close_df['Shift'].apply(lambda x: 0 if x > 0 else x)
            sd_1 = adj_close_df['Shift'].std()
            return_1 = (adj_close_df['Return'].iloc[-1] - adj_close_df['Return'].iloc[0])/100
            sr_1 = (return_1-0.07)/sd_1
            st.write('Sortino Ratio of Portfolio:', sr_1)

            benchdata['Shift'] = benchdata['Return'] - benchdata['Return'].shift(1)
            benchdata['Shift'] = benchdata['Shift'].fillna(0)
            benchdata['Shift'] = benchdata['Shift'].apply(lambda y: 0 if y > 0 else y)
            sd_2 = benchdata['Shift'].std()
            return_2 = (benchdata['Return'].iloc[-1] - benchdata['Return'].iloc[0])/100
            sr_2 = (return_2-0.07)/sd_2
            #st.write(return_2)
            #st.write(sd_2)
            st.write('Sortino Ratio of Benchmark:', sr_2)

            st.markdown('<p style="font-size:20px;"><b>Information Ratio</b></p>', unsafe_allow_html=True)
            adj_close_df_beta_bench['Diff'] = adj_close_df_beta['PV_Port'] - adj_close_df_beta_bench['PV_Bench']
            a = adj_close_df_beta_bench['Diff'].iloc[-1] - adj_close_df_beta_bench['Diff'].iloc[0]
            sd_pr_br = adj_close_df_beta_bench['Diff'].std()
            ir = a/sd_pr_br
            ir_formatted = "{:.10f}".format(ir)
            st.write('Information Ratio of Portfolio:', ir_formatted)

            st.markdown('<p style="font-size:20px;"><b>Sharpe Ratio</b></p>', unsafe_allow_html=True)
            sharpe_ratio_port = (return_1-0.07)/std_port #Risk Free rate is taken to be 7%
            sharpe_ratio_bench = (return_2-0.07)/std_bench #Risk free rate is taken to be 7%
            #st.write(return_1)
            #st.write(std_port)
            st.write('Sharpe ratio of Portfolio:',sharpe_ratio_port*100)
            #st.write(return_2)
            #st.write(std_bench)
            st.write('Sharpe ratio of Benchmark:',sharpe_ratio_bench*100)


            #P/E Ratio
            st.markdown('<p style="font-size:20px;"><b>P/E</b></p>', unsafe_allow_html=True)
            df['Earnings'] = df.apply(lambda row: (row['Market Cap(cr)']/row['PE']), axis=1)
            Total_Weighted_Mcap = ((df['Weightage(%)']/100) * df['Market Cap(cr)']).sum()
            df['Weighted_Earnings'] = (df['Weightage(%)'] / 100) * df['Earnings']
            Total_Earnings_weighted = df['Weighted_Earnings'].sum()
            #st.write(df.set_index('Company name'))
            #st.text("Total Weighted MarketCap(cr) is:")
            #st.write(Total_Weighted_Mcap)
            #st.text("Total Earnings(cr) is: ")
            #st.write(Total_Earnings_weighted)
            Port_PE_ratio=Total_Weighted_Mcap/Total_Earnings_weighted
            st.markdown('**Portfolio P/E ratio is:**')
            st.write(Port_PE_ratio)

            #P/B Ratio
            st.markdown('<p style="font-size:20px;"><b>P/B</b></p>', unsafe_allow_html=True)
            df['Number_of_Shares'] = df['CurrentMC'] / df['Price']
            df['Book Value'] = df.apply(lambda row: (row['Price']/row['PB']), axis=1)
            df['Total_Book_Value'] = df['Book Value'] * df['Number_of_Shares']
            df['Weighted_Book_Value'] = df['Weightage(%)'] * df['Total_Book_Value']
            Total_Weighted_BV = df['Weighted_Book_Value'].sum()
            Total_Weighted_Mcap = df['Weightage(%)'].dot(df['CurrentMC'])  # Weighted sum of Market Cap
            #st.write(df)
            Port_PB_ratio = Total_Weighted_Mcap / Total_Weighted_BV
            st.markdown('**Portfolio P/B ratio is:**')
            st.write(Port_PB_ratio)

            #st.markdown("**ROE(%) of Portfolio is:**")
            st.markdown('<p style="font-size:20px;"><b>ROE(%) of Portfolio is:</b></p>', unsafe_allow_html=True)
            roe_port = Port_PB_ratio / Port_PE_ratio
            st.write(roe_port)

            #st.markdown("**Dividend Yield:**")
            st.markdown('<p style="font-size:20px;"><b>Dividend Yield</b></p>', unsafe_allow_html=True)
            #PE_frame['DivLastYear'] = PE_frame.apply(lambda row:(row['Div Yield(%)'] * row['CurrentMC']), axis=1)
            #div_yield = PE_frame['DivLastYear'].sum() / Total_Weighted_Mcap
            #st.write(div_yield)
            #df['ADS'] = df.apply(lambda row:(row['DivPayOut']/100)*(row['EPSM22']+row['EPSJ22']+row['EPSS22']+row['EPSD22']), axis=1)
            df['ADS'] = df.apply(lambda row:(row['DivPayOut']/100)*(row['EPS_Annual']), axis=1)
            df['CalDivYield'] = df.apply(lambda row:row['ADS']*100/row['CurrentSP'],axis=1)
            df['Weighted_Div_Yield'] = df['Weightage(%)'] * df['CalDivYield']
            Total_Weighted_Div_Yield = (df['Weighted_Div_Yield'].sum())/100
            #st.write(df.set_index('Stock'))
            st.markdown("**Dividend yield(%) of the portfolio is:**")
            st.write(Total_Weighted_Div_Yield)

            #Return of stocks (Benchmark)
            stock_names_benchmark = price_data.columns[1:]
            benchmark_return_stocks = {}

            for stock in stock_names_benchmark:
                first_price = price_data[stock].iloc[0]  # First day's price
                last_price = price_data[stock].iloc[-1]  # Last day's price
                stock_return = (last_price - first_price) / 100
                benchmark_return_stocks[stock] = stock_return

            #st.write('benchmark return of stocks')
            #st.write(benchmark_return_stocks)

            #return of stocks (Portfolio)
            portfolio_stocks_data = pd.read_csv('adj_close_df.csv')
            stock_names_portfolio = portfolio_stocks_data.columns[1:]
            #st.write(stock_names_portfolio)
            portfolio_return_stocks = {}

            for stock in stock_names_portfolio:
                first_price = portfolio_stocks_data[stock].iloc[0]  # First day's price
                last_price = portfolio_stocks_data[stock].iloc[-1]  # Last day's price
                stock_return = (last_price - first_price) / 100
                portfolio_return_stocks[stock] = stock_return

            #st.write('portfolio return of stocks')
            #st.write(portfolio_return_stocks)


            #creating an excel ----- code -----


            ####################### Added code

            # stock_name_to_ticker = dict(zip(allocation_data['Company name'], allocation_data['SECURITY_ID']))
            # #st.write(stock_name_to_ticker)
            # ticker_name_to_stock = dict(zip(allocation_data['SECURITY_ID'], allocation_data['Company name']))


            # def assign_sectors(contrib_df, sector_data):
            #     sector_data["Company name"] = sector_data["Company name"].str.lower()
            #     stock_to_sector = dict(zip(sector_data["Company name"], sector_data["sector"]))

            #     # Map sectors to the companies in contrib_df
            #     contrib_df["Sector"] = contrib_df["Company"].map(stock_to_sector).fillna('Uncategorized')

            #     return contrib_df
            
            # sector_mapping_benchmark = pd.read_excel("./benchmarks/nifty50Benchmark.xlsx")
            # sector_mapping = pd.read_excel("./tempHoldingFile.xlsx")

            # df_port = assign_sectors(df_port, sector_mapping)

            # top_contributors['Company'] = top_contributors['Company'].map(ticker_name_to_stock)
            # top_contributors["Company"] = top_contributors["Company"].str.lower()

            # top_contributors = assign_sectors(top_contributors, sector_mapping_benchmark)

            # cont_ret_port_sector = df_port.groupby('Sector')['Contribution to return(%)'].sum()

            # cont_ret_port_sector = cont_ret_port_sector.reset_index()

            # cont_ret_port_sector['Contribution to return(%)'] *= 100

            # cont_ret_bench_sector = top_contributors.groupby('Sector')['Contribution to return(%)'].sum()

            # cont_ret_bench_sector = cont_ret_bench_sector.reset_index()

            #cont_ret_bench_sector['Contribution to return(%)'] *= 100







            #######################################

            portfolio_data = {
            "Sector": list(keys_axis),
            "Weight (%)": list(values_sector_axis)}

            benchmark_data = {
            "Sector": list(keys),
            "Weight (%)": list(values_sector)}

            portfolio_df_excel = pd.DataFrame(portfolio_data)
            benchmark_df_excel = pd.DataFrame(benchmark_data)

            # Create a unified sector list
            all_sectors_sorted = sorted(set(portfolio_df_excel["Sector"]).union(set(benchmark_df_excel["Sector"])))


            ############################# Reindex cont_ret_port_sector
            # cont_ret_port_sector = cont_ret_port_sector.set_index("Sector").reindex(all_sectors_sorted, fill_value=0).reset_index()
            # cont_ret_bench_sector = cont_ret_bench_sector.set_index("Sector").reindex(all_sectors_sorted, fill_value=0).reset_index()



            # Reindex Portfolio and Benchmark DataFrames
            portfolio_df_excel = portfolio_df_excel.set_index("Sector").reindex(all_sectors_sorted, fill_value=0).reset_index()
            benchmark_df_excel = benchmark_df_excel.set_index("Sector").reindex(all_sectors_sorted, fill_value=0).reset_index()

            # Rename columns for clarity
            portfolio_df_excel.columns = ["Sector", "Avg % Wgt (Port)"]
            benchmark_df_excel.columns = ["Sector", "Avg % Wgt (Bench)"]

            # Merge the two DataFrames on "Sector"
            aligned_df = pd.merge(portfolio_df_excel, benchmark_df_excel, on="Sector")

            df_allocation_effect_full = aligned_df[["Sector"]].merge(
                df_allocation_effect, on="Sector", how="left").fillna(0)
            
            df_selection_effect_full = aligned_df[["Sector"]].merge(
                df_selection_effect, on="Sector", how="left").fillna(0)
            
            aligned_df["Allocation Effect(%)"] = df_allocation_effect_full["Allocation Effect(%)"]
            aligned_df["Selection Effect(%)"] = df_selection_effect_full["Selection Effect(%)"]

            df_result["Contribution to Return (Port)"] = [row["Portfolio Weight"] * returns_of_portfolio_sector.get(row["Sector"], 0) / 100 for _, row in df_result.iterrows()]
            df_result["Contribution to Return (Bench)"] = [row["Benchmark Weight"] * sector_returns_dict.get(row["Sector"], 0) / 100 for _, row in df_result.iterrows()]

            df_result["Portfolio Return"] = [returns_of_portfolio_sector.get(sector, 0) for sector in df_result["Sector"]]
            df_result["Benchmark Return"] = [sector_returns_dict.get(sector, 0) for sector in df_result["Sector"]]
            df_result = df_result.reset_index(drop=True)
            #st.write(df_result)


            print("aligned_df")
            print(aligned_df)

            print("df_result")
            print(df_result)

            start_date = pd.to_datetime(start_date)
            
            summary_data = [
            ["Benchmark", name],
            ["Start Date", start_date],
            ["End Date", end_date],
            ["Currency", "INR"]]


            # # Align the lists by padding the shorter one with zeros
            # max_len = max(len(portfolio_weights), len(benchmark_weights))
            # portfolio_weights += [0] * (max_len - len(portfolio_weights))
            # benchmark_weights += [0] * (max_len - len(benchmark_weights))


            table_data = {
            "Sector": benchmark_df_excel["Sector"],
            "Avg % Wgt (Port)": aligned_df['Avg % Wgt (Port)']*100,
            "Avg % Wgt (Bench)": aligned_df['Avg % Wgt (Bench)'],
            "Avg wgt (+/-)": np.subtract(aligned_df['Avg % Wgt (Port)']*100, aligned_df['Avg % Wgt (Bench)']).tolist(),
            "Total Return (Port)": df_result["Portfolio Return"],
            "Total Return (Bench)": df_result["Benchmark Return"],
            "Total Return (+/-)": np.subtract(df_result["Portfolio Return"],df_result["Benchmark Return"]).tolist(),
            "Contribution to Return (Port)": df_result["Contribution to Return (Port)"],
            #"Contribution to Return (Port)": cont_ret_port_sector["Contribution to return(%)"],
            "Contribution to Return (Bench)": df_result["Contribution to Return (Bench)"], 
            #"Contribution to Return (Bench)": cont_ret_bench_sector["Contribution to return(%)"],
            "Contribution to Return (+/-)": np.subtract(df_result["Contribution to Return (Port)"], df_result["Contribution to Return (Bench)"]).to_list(),
            #"Contribution to Return (+/-)": np.subtract(cont_ret_port_sector["Contribution to return(%)"], cont_ret_bench_sector["Contribution to return(%)"]).to_list(),
            "Allocation Effect": aligned_df["Allocation Effect(%)"],
            "Selection Effect": aligned_df["Selection Effect(%)"],
            "Tot Atr": aligned_df["Allocation Effect(%)"] + aligned_df["Selection Effect(%)"]}
            #"Contribution to Return (Port)": [22.79, 4.63, 1.83, -0.19, 8.76, 5.55, 2.20, 0.00, 1.11, 4.09, 1.73, 1.91],
            #"Contribution to Return (Bench)": [26.91, 2.55, 3.77, 1.59, 3.42, 5.62, 1.12, 0.00, -1.28, -4.56, -45.49, -101.08],
            #"Contribution to Return (+/-)": [-4.13, 2.53, -2.27, -1.94, 6.14, 0.04, 1.25, 0.00, -1.28, -4.56, -1.92, -2.12],
            #"Allocation Effect (%)": [3.11, -1.13, 3.28, -0.69, 4.23, 0.50, 0.43, 0.00, 0.78, -2.22, 0.00, 0.03],
            #"Selection Effect (%)": [-7.24, -0.83, -0.06, -3.73, -1.36, -1.26, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
            #"Currency Effect (%)": [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
            #"Tot Attr": [-4.13, -1.95, 3.22, -4.43, 2.87, -0.76, 0.43, 0.03, 0.78, -2.22, 0.00, 0.03]}

            # Convert table data to a DataFrame
            table_df = pd.DataFrame(table_data)

            # Create the Excel file in memory
            output = io.BytesIO()
            workbook = xlsxwriter.Workbook(output, {'in_memory': True})
## Worksheet (Attribution Summary) code - Begin
        #     # worksheet = workbook.add_worksheet('Attribution Summary')

        #     # # Formatting objects
            bold_format = workbook.add_format({'bold': True, 'bg_color': '#D9EAD3'})
            header_format = workbook.add_format({'bold': True, 'align': 'center', 'bg_color': '#A6A6A6'})
            cell_format = workbook.add_format({'align': 'center'})
            date_format = workbook.add_format({'num_format': 'mm/dd/yyyy'})  # Date format
            left_align_bold_format = workbook.add_format({'bold': True, 'align': 'left'})  # Left-aligned bold for sectors

            ## Added decimal format of 2 decimal places
            decimal_format = workbook.add_format({'num_format': '0.00'})

        #     # # Write Summary section
        #     # # worksheet.write(0, 0, "Summary", bold_format)
        #     # # row = 1
        #     # # for key, value in summary_data:
        #     # #     worksheet.write(row, 0, key, bold_format)
        #     # #     worksheet.write(row, 1, value)
        #     # #     row += 1

        #     # worksheet.write(0, 0, "Summary", bold_format)
        #     # row = 1
        #     # for key, value in summary_data:
        #     #     worksheet.write(row, 0, key, bold_format)
        #     # # Check if the value is a datetime object and apply date formatting
        #     #     if isinstance(value, datetime):
        #     #         worksheet.write_datetime(row, 1, value, date_format)
        #     #     else:
        #     #         #worksheet.write(row, 1, value)
        #     #         worksheet.write(row, 1, value, decimal_format)
        #     #     row += 1

        #     # # Write table header
        #     # row += 1
        #     # worksheet.write(row, 0, "Attribution Summary (Grid)", bold_format)
        #     # row += 1
        #     # worksheet.write_row(row, 0, table_df.columns, header_format)


        # #     # Write table data

        # #   #  ############################### Original worksheet code
        # #     # for index, record in table_df.iterrows():
        # #     #     worksheet.write_row(row + 1 + index, 0, record.tolist(), cell_format)

        # #     ################### Added worksheet code
        # #     for index, record in table_df.iterrows():
        # #         for col_num, value in enumerate(record):
        # #             if isinstance(value, (int, float)):
        # #                 worksheet.write(row + 1 + index, col_num, value, decimal_format)  # Apply decimal format
        # #             else:
        # #                 worksheet.write(row + 1 + index, col_num, value, cell_format)

### Worksheet (Attribution Summary) code - end

            #Worksheet 2: Attribution to detail (New sheet)
            #worksheet1 = workbook.add_worksheet('Attribution Detail')
            
            def create_sector_stock_mapping(stock_dict, sector_mapping):
                sector_stock_dict = {}
                for stock in stock_dict.keys():
                    for sector, stocks in sector_mapping.items():
                        if stock in stocks:
                            if sector not in sector_stock_dict:
                                sector_stock_dict[sector] = []
                            sector_stock_dict[sector].append(stock)
                            break
                return sector_stock_dict

            def format_sector_stock_layout(data_dict):
                formatted_data = []
                for sector, stocks in data_dict.items():
                    formatted_data.append([sector])  # Add sector as a separate row
                    formatted_data.extend([[stock] for stock in stocks])  # Add each stock below
                return formatted_data
            
            # ################## Original sector_mapping
            # with open('stocks_by_sector.json', 'r') as file:
            #     sector_mapping = json.load(file)



            ################### Modified sector_mapping

            #sector_mapping_benchmark = pd.read_excel("./benchmarks/nifty50Benchmark.xlsx")
            if select_benchmark == 'NIFTY 50':
                sector_mapping_benchmark = pd.read_excel("./benchmarks/nifty50Benchmark.xlsx")
            if select_benchmark == 'NIFTY 100':
                sector_mapping_benchmark = pd.read_excel("./benchmarks/nifty100Benchmark.xlsx")
            if select_benchmark == 'NIFTY 500':
                sector_mapping_benchmark = pd.read_excel("./benchmarks/nifty500Benchmark.xlsx")

            if select_benchmark == 'SENSEX 50':
                sector_mapping_benchmark = pd.read_excel("./benchmarks/sensex50Benchmark.xlsx")
            
            if select_benchmark == 'BSE 100':
                sector_mapping_benchmark = pd.read_excel("./benchmarks/bse100Benchmark.xlsx")
            
            if select_benchmark == 'BSE 500':
                sector_mapping_benchmark = pd.read_excel("./benchmarks/bse500Benchmark.xlsx")

            
            
            


            sector_mapping = pd.read_excel("./tempHoldingFile.xlsx")
            


            # ############### Original get_sector_stocks()
            # def get_sector_stocks(sector, mapping):
            #     """Retrieve stocks belonging to a particular sector."""
            #     return mapping.get(sector, [])
            


            ################ Modified get_sector_stocks()
            def get_sector_stocks(sector, mapping):

                sector_stocks = mapping[mapping["sector"] == sector]["Company name"]

                return sector_stocks


            
            
            # Combine the data into a DataFrame
            data = []

            # Loop through all sectors in the benchmark
            for sector in sector_weights.keys():
            # Get sector weights
                portfolio_sector_weight = sector_weights_axis.get(sector, 0)
                benchmark_sector_weight = sector_weights.get(sector, 0)

            # Add sector weights to the data
                data.append([sector, round(portfolio_sector_weight, 2), round(benchmark_sector_weight, 2)])

            # Extract stocks for the sector
                sector_stocks = get_sector_stocks(sector, sector_mapping)

            # Add stock weights under each sector
                for stock in sector_stocks:
                    port_weight = round(portfolio_dict.get(stock, 0) * 100, 2)
                    bench_weight = round(benchmark_dict.get(stock, 0), 2)
                    if port_weight > 0 or bench_weight > 0:  # Include only stocks with weights
                        data.append([stock, port_weight, bench_weight])


            # Convert to DataFrame
            columns = ["Sector/Stock", "Avg % Wgt (Port)", "Avg % Wgt (Bench)"]
            weights_df = pd.DataFrame(data, columns=columns)
            weights_df["Avg wgt (+/-)"] = weights_df["Avg % Wgt (Port)"] - weights_df["Avg % Wgt (Bench)"]
            weights_df["Sector/Stock"] = weights_df["Sector/Stock"].fillna("Unknown").astype(str)

            print("weights_df")

            print(weights_df)

            sectors_list = list(sector_weights.keys())

            stock_name_to_ticker = dict(zip(allocation_data['Company name'], allocation_data['SECURITY_ID']))
            #st.write(stock_name_to_ticker)
            ticker_name_to_stock = dict(zip(allocation_data['SECURITY_ID'], allocation_data['Company name']))
            #st.write(ticker_name_to_stock)

            total_return_port = []
            total_return_bench = []

            for index, row in weights_df.iterrows():
                sector_or_stock = row["Sector/Stock"]
                # Check for sectors
                if sector_or_stock in sector_returns_dict:
                    ticker = sector_returns_dict[sector_or_stock]
                    #st.write(ticker)
                    total_return_port.append(returns_of_portfolio_sector.get(sector_or_stock, 0))
                    total_return_bench.append(sector_returns_dict.get(sector_or_stock, 0))
                    # total_return_port.append(returns_of_portfolio_sector[sector_or_stock])
                    # total_return_bench.append(sector_returns_dict.get(sector_or_stock, 0))
                # Check for stocks
                elif sector_or_stock in stock_name_to_ticker:
                    ticker = stock_name_to_ticker[sector_or_stock]
                    total_return_port.append(portfolio_return_stocks.get(ticker, 0))
                    total_return_bench.append(benchmark_return_stocks.get(ticker, 0))
                else:
                    total_return_port.append(0)
                    total_return_bench.append(0)

            weights_df["Total Return (Port)"] = total_return_port
            weights_df["Total Return (Bench)"] = total_return_bench
            weights_df["Total Return (+/-)"] = weights_df["Total Return (Port)"] - weights_df["Total Return (Bench)"]
            

            #contribution to return --- data into excel
            top_contributors['Company'] = top_contributors['Company'].map(ticker_name_to_stock)
            print("top_contributors")
            print(top_contributors)

            #top_contributors = top_contributors.reset_index(drop=True)



            ########################## Old assign_sectors()
            # def assign_sectors(contrib_df, sector_data):
            #     contrib_df['Sector'] = contrib_df['Company'].apply(
            #     lambda company: next((sector for sector, companies in sector_data.items() if company in companies), 'Uncategorized'))
            #     return contrib_df
            

            ############################ new assign_sectors()
            def assign_sectors(contrib_df, sector_data):
                sector_data["Company name"] = sector_data["Company name"].str.lower()
                stock_to_sector = dict(zip(sector_data["Company name"], sector_data["sector"]))

                # Map sectors to the companies in contrib_df
                contrib_df["Sector"] = contrib_df["Company"].map(stock_to_sector).fillna('Uncategorized')

                return contrib_df


            top_contributors["Company"] = top_contributors["Company"].str.lower()
            top_contributors = assign_sectors(top_contributors, sector_mapping_benchmark)

            print("top contributors after function")
            print(top_contributors)
            print("df_port")
            print(df_port)
            df_port = assign_sectors(df_port, sector_mapping)

            print("df_port after function")
            print(df_port)

        
            total_cont_port = []
            total_cont_bench = []

            cont_ret_port_sector = df_port.groupby('Sector')['Contribution to return(%)'].sum()

            print("cont_ret_port_sector")
            print(cont_ret_port_sector)
            cont_ret_port_sector_dict = cont_ret_port_sector.to_dict() # portfolio contribution of sector

            cont_ret_bench_sector = top_contributors.groupby('Sector')['Contribution to return(%)'].sum()
            print("cont_ret_bench_sector")
            print(cont_ret_bench_sector)
            cont_ret_bench_sector_dict = cont_ret_bench_sector.to_dict() # benchmark contribution of sector

            cont_ret_port_stock = dict(zip(df_port['Company'], df_port['Contribution to return(%)']))
            #st.write(cont_ret_port_stock)
            cont_ret_bench_stock = dict(zip(top_contributors['Company'], top_contributors['Contribution to return(%)']))
            #st.write(cont_ret_bench_stock)

            cont_ret_port_stock = {
                stock_name_to_ticker[name]: value
                for name, value in cont_ret_port_stock.items()
                if name in stock_name_to_ticker}
            #st.write(cont_ret_port_stock)
            # Map stock names to tickers for cont_ret_bench_stock
            cont_ret_bench_stock = {
                    stock_name_to_ticker[name]: value
                    for name, value in cont_ret_bench_stock.items()
                    if name in stock_name_to_ticker}
            #st.write(cont_ret_bench_stock)

            for index, row in weights_df.iterrows():
                sector_or_stock = row["Sector/Stock"]
                # Check for sectors
                if sector_or_stock in cont_ret_bench_sector_dict:
                    ticker = cont_ret_bench_sector_dict[sector_or_stock]
                    total_cont_port.append(cont_ret_port_sector_dict.get(sector_or_stock, 0))
                    total_cont_bench.append(cont_ret_bench_sector_dict.get(sector_or_stock, 0))
                #Check for stocks
                elif sector_or_stock in stock_name_to_ticker:
                    ticker = stock_name_to_ticker[sector_or_stock]
                    #st.write(ticker)
                    total_cont_port.append(cont_ret_port_stock.get(ticker, 0))
                    total_cont_bench.append(cont_ret_bench_stock.get(ticker,0))
                else:
                    total_cont_port.append(0)
                    total_cont_bench.append(0)

            weights_df["Contribution to Return (Port)"] = total_cont_port
            #weights_df["Contribution to Return (Port)"]  = weights_df["Contribution to Return (Port)"] * 100
            weights_df["Contribution to Return (Port)"]  = weights_df["Contribution to Return (Port)"]
            weights_df["Contribution to Return (Bench)"] = total_cont_bench
            weights_df["Contribution to Return (+/-)"] = weights_df["Contribution to Return (Port)"] - weights_df["Contribution to Return (Bench)"]

            allocation_effect_dict = dict(zip(df_allocation_effect['Sector'], df_allocation_effect['Allocation Effect(%)']))
            selection_effect_dict = dict(zip(df_selection_effect['Sector'], df_selection_effect['Selection Effect(%)']))

            alloc_effect = []
            selec_effect = []

            for index, row in weights_df.iterrows():
                sector_or_stock = row["Sector/Stock"]
                # Check for sectors
                if sector_or_stock in selection_effect_dict:
                    ticker = selection_effect_dict[sector_or_stock]
                    alloc_effect.append(allocation_effect_dict.get(sector_or_stock,0))
                    selec_effect.append(selection_effect_dict.get(sector_or_stock, 0))
                else:
                    alloc_effect.append(0)
                    selec_effect.append(0)
            
            weights_df["Allocation Effect"] = alloc_effect
            weights_df["Selection Effect"] = selec_effect
            weights_df["Tot Atr"] = weights_df["Allocation Effect"] + weights_df["Selection Effect"]

            
            # Display the DataFrame
            #st.write(weights_df)

            worksheet1 = workbook.add_worksheet('Attribution Detail')
            bold_left_format = workbook.add_format({'bold': True, 'align': 'left'})
            bold_center_format = workbook.add_format({'bold': True, 'align': 'center', 'bg_color': '#A6A6A6'})
            center_format = workbook.add_format({'align': 'center'})
            bold_summary_format = workbook.add_format({'bold': True, 'bg_color': '#D9EAD3', 'align': 'left'})


            ## Added decimal format of 2 decimal places
            decimal_format = workbook.add_format({'num_format': '0.00'})

            worksheet1.write(0, 0, "Summary", bold_summary_format)
            row = 1
            # for key, value in summary_data:
            #     worksheet1.write(row, 0, key, bold_left_format)
            #     worksheet1.write(row, 1, value, center_format)
            #     row += 1
            for key, value in summary_data:
                worksheet1.write(row, 0, key, bold_format)
            # Check if the value is a datetime object and apply date formatting
                if isinstance(value, datetime):
                    worksheet1.write_datetime(row, 1, value, date_format)
                else:
                    worksheet1.write(row, 1, value)
                row += 1

            row += 1
            worksheet1.write(row, 0, "Attribution Detail (All Securities, Including Buckets)", bold_summary_format)
            row += 1
            worksheet1.write_row(row, 0, table_df.columns, header_format)
            row += 1


            print("weights_df")
            print(weights_df)

            for index, record in weights_df.iterrows():
                sector_or_stock = record["Sector/Stock"]
                if sector_or_stock in sector_returns_dict:
                    worksheet1.write(row + index, 0, sector_or_stock, bold_left_format)
                    for col_num, value in enumerate(record[1:]):
                        #worksheet1.write(row + index, col_num + 1, value, center_format)
                        #worksheet1.write(row + index, col_num + 1, value, center_format, decimal_format)
                        worksheet1.write(row + index, col_num + 1, value, decimal_format)
                else:
                    for col_num, value in enumerate(record):
                        #worksheet1.write(row + index, col_num, value, center_format)
                        #worksheet1.write(row + index, col_num, value, center_format, decimal_format)
                        worksheet1.write(row + index, col_num, value, decimal_format)

            #worksheet1.set_column('A:A', 30)  # Sector/Stock
            #worksheet1.set_column('B:H', 15)  # Other numeric columns

            # Close workbook
            workbook.close()
            output.seek(0)

            # Streamlit download button
            st.download_button(
                label="ETE(Export to Excel)",
                data=output,
                file_name="Attribution_Report.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            



            ###################### session-code

            st.session_state["show_rebalancing_button"] = True


            
    else: 
        st.write("Please upload the Excel files to proceed.")




import math
#if st.button('Rebalancing'):
def rebalancing_button():
    #st.write('Yes, I am Working')
    if uploaded_files:
        for uploaded_file in uploaded_files:
            if uploaded_file.name == selected_file:









                df = pd.read_excel("tempHoldingFile.xlsx")

                global end_date
                global start_date


                #df = pd.read_excel(uploaded_file, usecols=['SECURITY_ID', 'QUANTITY','Date', 'Company name', '16 Sectors', 'SEBI Mcap Class'])
                df.dropna(inplace=True)  # Remove rows with NaN values
                df['Date'] = pd.to_datetime(df['Date'])
                #start_date = df['Date'].min()

                ## Dynamic start date
                start_date = start_date
                start_date = max(start_date, fixed_start_date)




                #global end_date
                end_date = pd.to_datetime(end_date)
                start_date = pd.to_datetime(start_date)
                stock_symbols = df['SECURITY_ID'].tolist()
                stock_names = []
                for symbol in stock_symbols:
                    ticker = yf.Ticker(symbol)
                    stock_name = ticker.info['longName']
                    stock_names.append(stock_name)
                df['yfName'] = stock_names
                num_stocks = df['QUANTITY']
                #st.write(num_stocks.reset_index(drop=True))
                #st.write(num_stocks)
                optimal_stocks_to_buy = dict(zip(df['SECURITY_ID'], df['QUANTITY']))
                #st.write(optimal_stocks_to_buy)
                historical_data = {}
                for symbol in stock_symbols:
                    stock_data = yf.download(symbol, start=start_date, end=end_date, multi_level_index=False)['Close']
                    historical_data[symbol] = stock_data
            
                adj_close_df = pd.DataFrame(historical_data)
                adj_close_df.to_csv('adj_close_df.csv')

                working_days = np.busday_count(start_date.date(), end_date.date())
                first_row_adj_close = adj_close_df.iloc[0]
                total_budget = (first_row_adj_close * df['QUANTITY'].values).sum()
                #st.write(total_budget)

                if select_benchmark == 'NIFTY 50':
                    csv_file = "./benchmarks/nifty50Benchmark.csv"
                    benchdata = yf.download("^NSEI", start = start_date, end = end_date, multi_level_index=False)['Close']
                    sectorbenchmark = pd.read_csv('./benchmarks/nifty50SectorWeightages.csv')

                if select_benchmark == 'NIFTY 500':
                    csv_file = "./benchmarks/nifty500Benchmark.csv"
                    benchdata = yf.download("^CRSLDX", start = start_date, end = end_date, multi_level_index=False)['Close']
                    sectorbenchmark = pd.read_csv('./benchmarks/nifty500SectorWeightages.csv')


                if select_benchmark == 'NIFTY 100':
                
                    csv_file = "./benchmarks/nifty100Benchmark.csv"
                    benchdata = yf.download("^CNX100", start = start_date, end = end_date, multi_level_index=False)['Close']
                    sectorbenchmark = pd.read_csv('./benchmarks/nifty100SectorWeightages.csv')
                    #name = 'NIFTY 100'

                
                # if select_benchmark == 'SENSEX 50':
                
                #     csv_file = "./benchmarks/sensex50Benchmark.csv"
                #     benchdata = yf.download("SNSX50.BO", start = start_date, end = end_date)['Adj Close']
                #     sectorbenchmark = pd.read_csv('./benchmarks/sensex50SectorWeightages.csv')
                #     name = 'SENSEX 50'

                if select_benchmark == 'BSE 100':
                
                    csv_file = "./benchmarks/bse100Benchmark.csv"
                    benchdata = yf.download("BSE-100.BO", start = start_date, end = end_date, multi_level_index=False)['Close']
                    sectorbenchmark = pd.read_csv('./benchmarks/bse100SectorWeightages.csv')
                    #name = 'BSE 100'

                if select_benchmark == 'BSE 500':
                
                    csv_file = "./benchmarks/bse500Benchmark.csv"
                    benchdata = yf.download("BSE-500.BO", start = start_date, end = end_date, multi_level_index=False)['Close']
                    sectorbenchmark = pd.read_csv('./benchmarks/bse500SectorWeightages.csv')
                    #name = 'BSE 500'
                # if select_benchmark == 'BSE 100':
                #     csv_file = "./benchmarks/nifty500Benchmark.csv"
                #     benchdata = yf.download("^CRSLDX", start = start_date, end = end_date)['Adj Close']
                #     sectorbenchmark = pd.read_csv('./benchmarks/nifty500SectorWeightages.csv')
                # elif select_benchmark == 'ICICI Prudential Nifty 100 ETF Fund':
                #     csv_file = "ICICIBenchmark.csv"
                #     benchdata = yf.download("ICICINF100.NS", start = start_date, end = end_date)['Adj Close']
                #     sectorbenchmark = pd.read_csv('ICICIBenchmark.csv')














                # if select_benchmark == 'NIFTY 50':
                #     csv_file = "NIFTY 50 Stock Weightages.csv"
                #     benchdata = yf.download("^NSEI", start = start_date, end = end_date)['Adj Close']
                #     sectorbenchmark = pd.read_csv('NIFTY 50 Sector Weightages.csv')
                # if select_benchmark == 'NSE 500':
                #     csv_file = "NSE 500 Stock Weightages.csv"
                #     benchdata = yf.download("^CRSLDX", start = start_date, end = end_date)['Adj Close']
                #     sectorbenchmark = pd.read_csv('NSE 500 Sector Weightages.csv')
                # elif select_benchmark == 'ICICI Prudential Nifty 100 ETF Fund':
                #     csv_file = "ICICIBenchmark.csv"
                #     benchdata = yf.download("ICICINF100.NS", start = start_date, end = end_date)['Adj Close']
                #     sectorbenchmark = pd.read_csv('ICICIBenchmark.csv')







                def build_bqm(alpha, _mu, _sigma, cardinality):  ############ Error in providing the threshold and cardinality. Go to function calling
                    n = len(_mu)
                    mdl = Model(name="stock_selection")
                    x = mdl.binary_var_list(range(n), name="x")
                    objective = alpha * (x @ _sigma @ x) - _mu @ x
                    # cardinality constraint
                    mdl.add_constraint(mdl.sum(x) == cardinality)
                    mdl.minimize(objective)
                    qp = from_docplex_mp(mdl)
                    qubo = QuadraticProgramToQubo().convert(qp)
                    bqm = dimod.as_bqm(
                        qubo.objective.linear.to_array(),
                        qubo.objective.quadratic.to_array(),
                        dimod.BINARY,)
                    return bqm
                
                constituents = pd.read_csv(csv_file)
                tickers = constituents['SECURITY_ID'].to_list()
                #st.write(tickers)
                use_local = False
                if use_local is False:
                        #end_date = end_date
                        #start_date = start_date
                        adj_close_df_1 = pd.DataFrame()
                        for ticker in tickers:
                            data = yf.download(ticker, start=start_date, end=end_date, multi_level_index=False)['Close']
                            adj_close_df_1[ticker] = data
                        adj_close_df_1.to_csv('benchmark.csv')
                
                #first_row_adj_close = adj_close_df.iloc[0]
                #total_budget = (first_row_adj_close * df['QUANTITY'].values).sum()
                #st.write(total_budget)

                def process_portfolio(init_holdings, total_budget):
                    st.write("init_holdings at the start of the function")
                    st.write(init_holdings)

                    cfg.hpfilter_lamb = 6.25
                    cfg.q = 1.0  # risk-aversion factor
                    # classical
                    cfg.fmin = 0.01  # 0.001
                    cfg.fmax = 0.5  # 0.5
        
                    constituents = pd.read_csv(csv_file)
                    tickers = constituents['SECURITY_ID'].to_list()
                    st.write("tickers")
                    st.write(tickers)

                    data = pd.read_csv('benchmark.csv', parse_dates=['Date'])
                    sector_map = constituents.loc[constituents['SECURITY_ID'].isin(tickers)]
                    st.write("sector_map")
                    st.write(sector_map)
                    #st.write(sector_map)
                    dates = data["Date"].to_numpy()
                    monthly_df = data.resample('3M', on='Date').last() # resample to every 3 months
                    month_end_dates = monthly_df.index
                    st.write("month_end_dates")
                    st.write(month_end_dates)
                    available_sectors, counts = np.unique(np.array(sector_map.sector.tolist()), return_counts=True)
                    #total_budget = 537787.2409820557 #Make the budget dynamic
                    #global total_budget
                    
                    total_budget = total_budget
                    print(total_budget)
                    num_months = len(month_end_dates)
                    first_purchase = True 
                    result = {}
                    update_values = [0]
                    months = []
                    start_month = 0
                    headers = ['Date', 'Value'] + list(tickers) + ['Risk', 'Returns', 'SR']
                    opt_results_df = pd.DataFrame(columns=headers)
                    row = []
                    tickers = np.array(tickers)
                    wallet = 0.0

                    for i, end_date in enumerate(month_end_dates[start_month:]):
                        df = data[dates <= end_date].copy()
                        st.write("df")
                        st.write(df)

                        df.set_index('Date', inplace=True)
                        months.append(df.last_valid_index().date())
                        if first_purchase:
                            budget = total_budget
                            initial_budget = total_budget
                        else:
                            value = sum([df.iloc[-1][s] * init_holdings.get(s, 0) for s in tickers]) # portfolio
                            #print(i, f"Portfolio : {budget:.2f},")
                            #print(f"Profit: {budget - initial_budget:.2f}")
                            update_values.append(budget - initial_budget)

                        for s in df.columns:
                            cycle, trend = hpfilter(df[s], lamb=cfg.hpfilter_lamb)
                            df[s] = trend

                        log_returns = np.log(df) - np.log(df.shift(1))
                        null_indices = np.where((log_returns.isna().sum() > 1).to_numpy())[0]
                        drop_stocks = df.columns[null_indices]
                        log_returns = log_returns.drop(columns=drop_stocks)
                        log_returns = log_returns.dropna()
                        tickers = log_returns.columns
                        st.write("tickers")
                        st.write(tickers)

                        mu = log_returns.mean().to_numpy() * working_days
                        sigma = log_returns.cov().to_numpy() * working_days
                        price = df.iloc[-1] # last day price

                        #Sell Idea
                        threshold = 4 # Sell all stocks for `threshold` companies
                        tickers_holding = np.array(list(init_holdings.keys())) # Names of the companies in initial holdings
                        indices = np.isin(tickers, tickers_holding) # Indices of `tickers_holding` in the list of all companies `tickers`
                        argsort_indices = np.argsort(mu[indices]) # Obtain `mu` values at `indices`. Sort it.
                        st.write("argsort_indices")
                        st.write(argsort_indices)

                        sell_indices =  argsort_indices < threshold # indices of the least `threshold` companies (least in terms of mu value)

                        st.write("sell_indices")
                        st.write(sell_indices)
                        sell_tickers = tickers_holding[argsort_indices][sell_indices] # names of those companies
                        st.write("sell_tickers")
                        st.write(sell_tickers)

                        sectors = sector_map.loc[sector_map['SECURITY_ID'].isin(sell_tickers)]['sector'].tolist()

                        sectors = set(sectors) # remove duplicates

                        tickers_new = sector_map.loc[sector_map['sector'].isin(sectors)]['SECURITY_ID'].tolist()
                        tickers_new = np.intersect1d(np.array(tickers_new), np.array(tickers))
                        tickers_new = np.setdiff1d(np.array(tickers_new), np.array(sell_tickers))
        
                        keep_indices = np.in1d(np.array(tickers), tickers_new)
                        mu_new = mu[keep_indices]
                        sigma_new = sigma[keep_indices][:, keep_indices]

                        sales_revenue = 0.0
                        for tick in sell_tickers:
                            sales_revenue += init_holdings[tick] * price[tick]
                            init_holdings.pop(tick, None) # remove that company from holdings
                        bqm = build_bqm(cfg.q, mu_new, sigma_new, threshold) ################ Fix the cardinality and threshold here
                        st.write(bqm)
                        sampler_sa = SimulatedAnnealingSampler()
                        result_sa = sampler_sa.sample(bqm, num_reads=5000)
                        selection = list(result_sa.first.sample.values())
                        selection = np.array(selection, dtype=bool)

                        tickers_selected = tickers_new[selection]
                        st.write("tickers_selected")
                        st.write(tickers_selected)

                        keep_indices = np.in1d(tickers_new, tickers_selected)
                        st.write("keep_indices")
                        st.write(keep_indices)

                        mu_selected = mu_new[keep_indices]
                        sigma_selected = sigma_new[keep_indices][:, keep_indices]

                        # st.write(init_holdings)
                        # st.write(tickers_selected)

                        qpo = SinglePeriod(cfg.q,
                                            mu_selected,
                                            sigma_selected,
                                            sales_revenue + wallet, 
                                            np.array([price[tick] for tick in tickers_selected]), 
                                            tickers_selected)
                        
                        solution = qpo.solve_cqm(init_holdings)
                        result = solution['stocks'].copy()
                        asset_weights = qpo._weight_allocation(solution['stocks'])
                        optimal_weights_dict = qpo._get_optimal_weights_dict(
                        asset_weights, solution['stocks'])
                        metrics = qpo._get_risk_ret(asset_weights) # risk, returns and sharpe ratio

                        for tick, val in result.items():
                            if val != 0:
                                #print(f"{tick}, ({sector_map.loc[sector_map['symbol'] == tick]['sector'].tolist()[0]})", ' '*2, val)
                                if tick not in init_holdings.keys():
                                    init_holdings[tick] = val
                                else:
                                    init_holdings[tick] += val
                        
                        value = sum([price[s] * result.get(s, 0.0) for s in tickers_new]) # Amount invested in purchasing
                        value_port = sum([price[s] * init_holdings.get(s, 0.0) for s in init_holdings]) # Portfolio Value after Rebalancing
                        wallet = (sales_revenue + wallet) - value # Amount left in wallet

                        returns = f"{metrics['returns']:.2f}"
                        risk = f"{metrics['risk']:.2f}"
                        sr = f"{metrics['sharpe_ratio']:.2f}"

                        tickers = constituents['SECURITY_ID'].to_list()
                        row = [months[-1].strftime('%Y-%m-%d'), value_port/initial_budget] + \
                            [init_holdings.get(s, 0) for s in tickers] + \
                            [risk, returns, sr]
                        
                        opt_results_df.loc[i] = row.copy()
                        first_purchase = False
                    return opt_results_df
                
                #st.write(optimal_stocks_to_buy)
                #optimal_stocks_to_buy = {'BHARTIARTL.NS': 109.0, 'HDFCBANK.NS': 92.0, 'HINDUNILVR.NS': 92.0, 'ICICIBANK.NS': 104.0, 'INFY.NS': 86.0, 'ITC.NS': 112.0, 'LT.NS': 118.0, 'RELIANCE.NS': 107.0, 'SBIN.NS': 104.0, 'TCS.NS': 95.0, 'BAJFINANCE.NS':100.0, 'MARUTI.NS': 87.0, 'TITAN.NS':60.0}
                process_portfolio_amar = process_portfolio(optimal_stocks_to_buy, total_budget)
                process_portfolio_amar_df = process_portfolio_amar.to_csv('rebalancing_test.csv')
                dataf = pd.read_csv('rebalancing_test.csv')
                #st.write(dataf)
                new_data_dict = optimal_stocks_to_buy
                new_row_df = pd.DataFrame(new_data_dict, index=[0])
                for column in dataf.columns:
                    if column not in new_row_df.columns:
                        new_row_df[column] = "" if column == "Date" else 0
                new_row_df = new_row_df[dataf.columns]
                updated_dataf = pd.concat([new_row_df, dataf], ignore_index=True)
                #st.write(updated_dataf)

                tickers = constituents['SECURITY_ID'].to_list()
                for column in updated_dataf.columns:
                    if column in tickers:
                        updated_dataf[column] = updated_dataf[column].diff().fillna(updated_dataf[column])
                updated_dataf = updated_dataf.drop('Unnamed: 0', axis=1)
                columns_to_style = tickers
                def apply_styling(value):
                    if isinstance(value, (int, float)):  # Ensure the value is numeric before comparison
                        if value > 0:
                            return f'<span style="color: green;">{value}</span>'
                        elif value < 0:
                            return f'<span style="color: red;">{value}</span>'
                        else:
                            return value
                    return value  # If the value is already a string, return it as-is

                #def apply_styling(value):
                    #if int(value) > 0:
                        #return f'<span style="color: green;">{value}</span>'
                    #elif int(value) < 0:
                        #return f'<span style="color: red;">{value}</span>'
                    #else:
                        #return value
                    
                for column in columns_to_style:
                    updated_dataf[column] = updated_dataf[column].apply(apply_styling)
                #st.write(total_budget)
                updated_dataf["Value"] = updated_dataf["Value"] * total_budget
                updated_dataf = updated_dataf.loc[:, (updated_dataf != 0).any(axis=0)] # to remove columns with 0s
                updated_dataf = updated_dataf.to_html(float_format=lambda x: '{:.2f}'.format(x), escape=False)
                #st.write(new_data_dict)
                st.write("Portfolio after Rebalancing:")
                st.write(updated_dataf, unsafe_allow_html=True)
                df_fta = pd.read_html(updated_dataf)[0]
                df_fta['Date'] = pd.to_datetime(df_fta['Date'])
                adj_close_df = adj_close_df.reset_index()
                adj_close_df.columns = adj_close_df.columns.str.strip()
                #st.write(adj_close_df)
                #st.write(adj_close_df['Date'])
                

                # df_fta = pd.read_html(updated_dataf)[0]
                # df_fta['Date'] = pd.to_datetime(df_fta['Date'], format='%Y-%m-%d')
                # dates_for_extraction = df_fta['Date'].tolist()
                # extracted_df = adj_close_df[adj_close_df['Date']].isin(dates_for_extraction)
                # temp_extracted_df = extracted_df.set_index('Date', inplace = True)
                # num_stocks = num_stocks.tolist()
                # portfolio_values = (temp_extracted_df*num_stocks).sum(axis=1)
                # st.write(portfolio_values)

                stock_quantities = dict(zip(df['SECURITY_ID'], df['QUANTITY']))

                # Now we will multiply the quantities with the adjusted close prices
                for stock, quantity in stock_quantities.items():
                    if stock in adj_close_df.columns:
                        adj_close_df[stock] = adj_close_df[stock] * quantity

                
                # Sum across all stock columns to calculate the total portfolio value
                adj_close_df['portfolio value'] = adj_close_df[list(stock_quantities.keys())].sum(axis=1)

                dates_to_extract = df_fta['Date'].to_list()

                # adj_close_df['Date'] = pd.to_datetime(adj_close_df['Date'], format='%Y-%m-%d')
                # adj_close_df['Date'] = pd.to_datetime(adj_close_df['Date'])

                filtered_data = adj_close_df[adj_close_df['Date'].isin(pd.to_datetime(dates_to_extract))]

                #st.write(filtered_data)
                #st.write(df_fta)

                df2_cleaned = df_fta.dropna(subset=['Date'])  # Drop rows with None in the Date column
                df2_cleaned = df2_cleaned[df2_cleaned['Value'] != 0]
                
                #st.write(df2_cleaned)
                #st.write(filtered_data)

                df2_cleaned['Date'] = pd.to_datetime(df2_cleaned['Date'])
                filtered_data['Date'] = pd.to_datetime(filtered_data['Date'])

                #st.write(df2_cleaned[['Date', 'Value']])
                #st.write(filtered_data[['Date', 'portfolio value']])


                min_df2_cleaned = df2_cleaned['Value'].min()
                max_df2_cleaned = df2_cleaned['Value'].max()

                min_filtered_data = filtered_data['portfolio value'].min()
                max_filtered_data = filtered_data['portfolio value'].max()

                minimum_value = min(min_df2_cleaned, min_filtered_data)

                maximum_value = max(max_df2_cleaned, max_filtered_data)

                #y_range = math.ceil(maximum_value / minimum_value )
                

                st.text('with rebalancing')
                st.write(df2_cleaned['Value'])
                st.text('without rebalancing')
                st.write(filtered_data['portfolio value'])

                def count_integer_digits(n):
                    return len(str(int(abs(n))))
                
                abs_value = 10 ** (count_integer_digits(minimum_value)-1)
                y_range = math.ceil(maximum_value / abs_value)

                #st.write(abs_value)

                # Plotting
                fig_rebalancing = go.Figure()

                # Add trace for rebalanced portfolio value (divided by 10^7 to convert to crores)
                fig_rebalancing.add_trace(go.Scatter(
                    x=df2_cleaned['Date'],
                    y=df2_cleaned['Value'],
                    mode='lines+markers',
                    name='Portfolio Value With Rebalancing',
                    line=dict(color='red'),
                    showlegend=True
                ))

                # Add trace for original portfolio value (divided by 10^7 to convert to crores)
                fig_rebalancing.add_trace(go.Scatter(
                    x=filtered_data['Date'],
                    y=filtered_data['portfolio value'],
                    mode='lines+markers',
                    name='Portfolio Value Without Rebalancing',
                    line=dict(color='blue'),
                    showlegend=True
                ))

# Set layout and axis ranges
                fig_rebalancing.update_layout(
                    title="Rebalanced Portfolio Values Over Time",
                    xaxis_title='Date', 
                    #yaxis_title=f"Portfolio Values scaled with Minimum portfolio value {round(minimum_value,2)}",
                    yaxis_title = "Portfolio Value",
                    autosize=False, 
                    width=1000, 
                    height=600,
                    #yaxis_range=[0.15, 0.22]  # Adjust this range according to your data
                    #yaxis_range = [0, y_range]
                    )

# Display the plot in Streamlit
                st.plotly_chart(fig_rebalancing)

    


                #st.write(adj_close_df)

                st.session_state["show_reset_button"] = True


                    
    else: 
        st.write("Please upload the Excel files to proceed.")
    
def handle_reset():
    st.session_state["show_rebalancing_button"] = False
    st.session_state["show_reset_button"] = False




##################### session-code

if st.button("Next"):
    next_button()

# UI for "Rebalancing" button
if st.session_state["show_rebalancing_button"]:
    if st.button("Rebalancing"):
        rebalancing_button()


###################### session-code

# if st.button("Reset Workflow"):
#     st.session_state["show_rebalancing_button"] = False

if st.session_state["show_reset_button"]:
    if st.button("Reset Workflow"):
        handle_reset()