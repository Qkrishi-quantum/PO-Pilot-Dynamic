import streamlit as st
import pandas as pd
from plotly import graph_objs as go
import plotly.express as px
from scipy.stats import linregress
from datetime import date
import yfinance as yf
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
from datetime import datetime

seed = 42
cfg = ml_collections.ConfigDict()

st.set_page_config(page_title="PilotProject", page_icon=":chart_with_upwards_trend:", layout="wide")
st.title("Pilot Project on Portfolio Optimisation")

uploaded_files = st.file_uploader("Choose Excel files", type=["xlsx", "xls"], accept_multiple_files=True)
file_names = [uploaded_file.name for uploaded_file in uploaded_files]
selected_file = st.selectbox("Select a file", file_names)
select_benchmark = st.selectbox("Select the benchmark", options=['NIFTY 50', 'NSE 500', 'ICICI Prudential Nifty 100 ETF Fund'])
end_date = st.date_input("Select end date", value=datetime(2024, 1, 31))
total_budget = 1

if st.button('Next'):
    if uploaded_files:
        for uploaded_file in uploaded_files:
            if uploaded_file.name == selected_file:
                df = pd.read_excel(uploaded_file, usecols=['SECURITY_ID', 'QUANTITY','Date', 'Company name', '16 Sectors', 'SEBI Mcap Class', 'PE', 'PB', 'Price', 'Div Yield(%)', 'DivPayOut', 'CurrentMC', 'CurrentSP', 'EPSM22', 'EPSJ22', 'EPSS22', 'EPSD22', 'Market Cap(cr)'])
                df.dropna(inplace=True)  # Remove rows with NaN values
                df['Date'] = pd.to_datetime(df['Date'])
                start_date = df['Date'].min()
                end_date = pd.to_datetime(end_date)
                stock_symbols = df['SECURITY_ID'].tolist()
                stock_names = []
                for symbol in stock_symbols:
                    ticker = yf.Ticker(symbol)
                    stock_name = ticker.info['longName']
                    stock_names.append(stock_name)
                df['yfName'] = stock_names
                #num_stocks = df['Quantity']
                #st.write(num_stocks.reset_index(drop=True))
                #st.write(num_stocks)
                historical_data = {}
                for symbol in stock_symbols:
                    historical_data[symbol] = yf.download(symbol, start=start_date, end=end_date)['Adj Close']
            
                adj_close_df = pd.DataFrame(historical_data)
                adj_close_df.to_csv('adj_close_df.csv')
                #st.write("Adjusted Closing Prices")

                st.write(adj_close_df)
         
            if select_benchmark == 'NIFTY 50':
                csv_file = "NIFTY 50 Stock Weightages.csv"
                benchdata = yf.download("^NSEI", start = start_date, end = end_date)['Adj Close']
                sectorbenchmark = pd.read_csv('NIFTY 50 Sector Weightages.csv')
            if select_benchmark == 'NSE 500':
                csv_file = "NSE 500 Stock Weightages.csv"
                benchdata = yf.download("^CRSLDX", start = start_date, end = end_date)['Adj Close']
                sectorbenchmark = pd.read_csv('NSE 500 Sector Weightages.csv')
            elif select_benchmark == 'ICICI Prudential Nifty 100 ETF Fund':
                csv_file = "ICICIBenchmark.csv"
                benchdata = yf.download("ICICINF100.NS", start = start_date, end = end_date)['Adj Close']
                sectorbenchmark = pd.read_csv('ICICIBenchmark.csv')

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

            first_row_adj_close = adj_close_df.iloc[0]
            total_budget = (first_row_adj_close * df['QUANTITY'].values).sum()
            wt_stock = ((first_row_adj_close * df['QUANTITY'].values) / total_budget) * 100
            #df['Weightage(%)'] = ((first_row_adj_close * df['QUANTITY'].values) / total_budget) * 100
            portfolio_dict = dict(zip(df['yfName'], wt_stock))
            df['Weightage(%)'] = portfolio_dict.values()
            #st.write(df)
            vp_list = list(portfolio_dict.values())
            portfolio_array = np.array(vp_list)

            rows = len(stock_symbols)
            wt_bench = pd.read_csv(csv_file, nrows=rows).reset_index(drop=True) #nrows should be dynamic
            wt_bench_filtered = wt_bench[['Company Name', 'Weightage(%)']]
            benchmark_dict = dict(zip(wt_bench_filtered.iloc[:, 0], wt_bench_filtered.iloc[:, 1]))
            bp_list = list(benchmark_dict.values())
            benchmark_array = np.array(bp_list)

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Avg % Wgt Portfolio:**")
                for stock, value in portfolio_dict.items():
                    percentage = round(value, 2)  # Multiply by 100 and round off to 2 decimal places
                    st.text(f"{stock:<45}{percentage:>15}")
    
                st.markdown("**Returns and Risk of Portfolio:**")
                mvo_miqp_bench = cpo.get_metrics(portfolio_array)
                for metric, value in mvo_miqp_bench.items():
                    if metric in ['returns', 'risk']:
                        display_value = round(value ,2)
                    else:
                        display_value = round(value, 2)
                    st.text(f"{metric:<45}{display_value:>15}")

            with col2:
                st.markdown("**Avg % Wgt Benchmark:**")
                for stock, value in benchmark_dict.items():
                    percentage = round(value, 2)
                    st.text(f"{stock:<35}{percentage:>15}")
                st.markdown("**Returns and Risk of Benchmark:**")
                mvo_miqp_bench = cpo.get_metrics(benchmark_array)
                for metric, value in mvo_miqp_bench.items():
                    if metric in ['returns', 'risk']:
                        display_value = round(value ,2)
                    else:
                        display_value = round(value, 2)
                    st.text(f"{metric:<35}{display_value:>15}")

            colors = ['gold', 'mediumturquoise', 'darkorange', 'lightgreen', 'DarkOrchid', 'DeepPink', 'Maroon', 'MistyRose', 'Olive', 'Salmon' ]
            fig = go.Figure(data=[go.Pie(labels=list(portfolio_dict.keys()), values=list(portfolio_dict.values()), hole=.3)])
            fig.update_traces(hoverinfo='label+percent',textfont_size=15, marker=dict(colors=colors, line=dict(color='#000000', width=2)))
            st.markdown("**Pie Chart of Stock Weights Allocated Portfolio:**")
            st.plotly_chart(fig)

            with open('stocks_by_sector.json', 'r') as file:
                data = json.load(file)
            
            sector_weights_axis= {}
            sectors = data
            for stock, weight in portfolio_dict.items():
                for sector, stocks_in_sector in sectors.items():
                    if stock in stocks_in_sector:
                        sector_weights_axis.setdefault(sector, 0)
                        sector_weights_axis[sector] += weight
            keys_axis = sector_weights_axis.keys()
            values_sector_axis = sector_weights_axis.values()
            fig_sector_axis = go.Figure(data=[go.Pie(labels=list(keys_axis),values=list(values_sector_axis), hole=.3)])
            fig_sector_axis.update_traces(hoverinfo='label+percent',textfont_size=15, marker=dict(line=dict(color='#000000', width=2)))
            st.markdown("**Pie Chart of Sector Weights Portfolio:**")
            st.plotly_chart(fig_sector_axis)

            fig_bench = go.Figure(data=[go.Pie(labels=list(benchmark_dict.keys()), values=list(benchmark_dict.values()), hole=.3)])
            fig_bench.update_traces(hoverinfo='label+percent',textfont_size=15, marker=dict(colors=colors, line=dict(color='#000000', width=2)))
            st.markdown("**Pie Chart of Stock Weights Allocated Benchmark:**")
            st.plotly_chart(fig_bench)

            sector_weights= {}
            sectors_bench = data
            for stock, weight in benchmark_dict.items():
                for sector, stocks_in_sector in sectors_bench.items():
                    if stock in stocks_in_sector:
                        sector_weights.setdefault(sector, 0)
                        sector_weights[sector] += weight

            keys = sector_weights.keys()
            values_sector = sector_weights.values()
            fig_sector = go.Figure(data=[go.Pie(labels=list(keys),values=list(values_sector), hole=.3)])
            fig_sector.update_traces(hoverinfo='label+percent',textfont_size=15, marker=dict(line=dict(color='#000000', width=2)))
            st.markdown("**Pie Chart of Sector Weights Benchmark:**")
            st.plotly_chart(fig_sector)

            st.markdown('<p style="font-size:20px;"><b>Line chart of Portfolio against the Benchmark (Rebased to 100 for initial date)</b></p>', unsafe_allow_html=True)
            quantity_dict = pd.Series(df.QUANTITY.values, index=df.SECURITY_ID).to_dict()
            for symbol in adj_close_df.columns[1:]:  # Skip the 'Date' column (index 0)
                if symbol in quantity_dict:
                    adj_close_df[symbol] = adj_close_df[symbol] * quantity_dict[symbol]
            adj_close_df['Portfolio Value'] = adj_close_df.iloc[:, 1:].sum(axis=1)
            adj_close_df['Return'] = (adj_close_df['Portfolio Value'] / adj_close_df['Portfolio Value'][0]) * 100
            benchdata = pd.DataFrame(benchdata)
            benchdata['Return'] = (benchdata['Adj Close']/benchdata['Adj Close'][0]) * 100

            fig_compare = go.Figure()
            fig_compare.add_trace(go.Scatter(x= adj_close_df.index, 
                    y=  adj_close_df['Return'],
                    mode='lines+markers', 
                    name='Return Portfolio', 
                    line=dict(color='red')))
            fig_compare.add_trace(go.Scatter(x=benchdata.index, 
                    y=benchdata['Return'], 
                    mode='lines+markers', 
                    name='Return Benchmark', 
                    line=dict(color='blue')))
            fig_compare.update_layout(title='Return Over Time',
                    xaxis_title='Date', 
                    yaxis_title='Return',
                    autosize=False, 
                    width=1000, 
                    height=600,)
            st.plotly_chart(fig_compare)

            st.markdown('<p style="font-size:20px;"><b>Portfolio Weight vs Benchmark Weight</b></p>', unsafe_allow_html=True)
            all_sectors = set(sector_weights.keys()).union(set(sector_weights_axis.keys()))

            # Create a DataFrame
            data = {
            "Sector": [],
            "Portfolio Weight": [],
            "Benchmark Weight": [],
            "Status": [],
            "+/-": []
            }

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

            df_port = pd.DataFrame(sorted_contribution_port, columns=['Company', 'Contribution to return(%)'])
            df_port = df_port.sort_values(by='Contribution to return(%)', ascending=False)

            cr_data = pd.read_csv(csv_file)
            cr_tickers = cr_data['Symbol'].tolist()
            cr_weightages = cr_data['Weightage(%)'].tolist()
            cr_dict = dict(zip(cr_tickers, cr_weightages))
            #st.write(cr_dict)
            working_days = np.busday_count(start_date_port.date(), end_date_port.date())

            def calculate_top_contributors(tickers_weights, start_date, end_date):
                # Download stock data for the given date range
                stock_data = yf.download(list(tickers_weights.keys()), start=start_date, end=end_date)['Adj Close']
    
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
            col5, col6 = st.columns(2)
            with col5:
                st.markdown('**Contribution to return - Portfolio, Top 10/Bottom 10**') 
                st.write(df_port.set_index('Company'))
            with col6:
                st.markdown('**Contribution to return - Benchmark, Top 10/Bottom 10**') 
                st.write(top_contributors.set_index('Company'))


            st.markdown('<p style="font-size:20px;"><b>Beta</b></p>', unsafe_allow_html=True)
            end_date = end_date
            start_date_beta = datetime(end_date.year - 2, end_date.month, end_date.day)
            historical_data_beta = {}
            for symbol in stock_symbols:
                historical_data_beta[symbol] = yf.download(symbol, start=start_date_beta, end=end_date)['Adj Close']

            adj_close_df_beta = pd.DataFrame(historical_data_beta)
            adj_close_df_beta.to_csv('adj_close_df_beta.csv')

            stock_wt = dict(zip(df['SECURITY_ID'], wt_stock))
            stock_quantities = dict(zip(df['SECURITY_ID'],df['QUANTITY']))

            portfolio_values_port = adj_close_df_beta.apply(lambda row: sum(
            row[stock] * stock_quantities[stock] for stock in stock_quantities), axis=1)
            adj_close_df_beta['PV_Port'] = portfolio_values_port

            st.write(adj_close_df_beta)

            historical_data_beta_bench = {}
            for symbol in cr_dict:
                historical_data_beta_bench[symbol] = yf.download(symbol, start=start_date_beta, end=end_date)['Adj Close']
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
            df['ADS'] = df.apply(lambda row:(row['DivPayOut']/100)*(row['EPSM22']+row['EPSJ22']+row['EPSS22']+row['EPSD22']), axis=1)
            df['CalDivYield'] = df.apply(lambda row:row['ADS']*100/row['CurrentSP'],axis=1)
            df['Weighted_Div_Yield'] = df['Weightage(%)'] * df['CalDivYield']
            Total_Weighted_Div_Yield = (df['Weighted_Div_Yield'].sum())/100
            #st.write(df.set_index('Stock'))
            st.markdown("**Dividend yield(%) of the portfolio is:**")
            st.write(Total_Weighted_Div_Yield)
            
    else: 
        st.write("Please upload the Excel files to proceed.")



import math
if st.button('Rebalancing'):
    #st.write('Yes, I am Working')
    if uploaded_files:
        for uploaded_file in uploaded_files:
            if uploaded_file.name == selected_file:
                df = pd.read_excel(uploaded_file, usecols=['SECURITY_ID', 'QUANTITY','Date', 'Company name', '16 Sectors', 'SEBI Mcap Class'])
                df.dropna(inplace=True)  # Remove rows with NaN values
                df['Date'] = pd.to_datetime(df['Date'])
                start_date = df['Date'].min()
                end_date = pd.to_datetime(end_date)
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
                historical_data = {}
                for symbol in stock_symbols:
                    historical_data[symbol] = yf.download(symbol, start=start_date, end=end_date)['Adj Close']
            
                adj_close_df = pd.DataFrame(historical_data)
                adj_close_df.to_csv('adj_close_df.csv')

                working_days = np.busday_count(start_date.date(), end_date.date())
                first_row_adj_close = adj_close_df.iloc[0]
                total_budget = (first_row_adj_close * df['QUANTITY'].values).sum()

                if select_benchmark == 'NIFTY 50':
                    csv_file = "NIFTY 50 Stock Weightages.csv"
                    benchdata = yf.download("^NSEI", start = start_date, end = end_date)['Adj Close']
                    sectorbenchmark = pd.read_csv('NIFTY 50 Sector Weightages.csv')
                if select_benchmark == 'NSE 500':
                    csv_file = "NSE 500 Stock Weightages.csv"
                    benchdata = yf.download("^CRSLDX", start = start_date, end = end_date)['Adj Close']
                    sectorbenchmark = pd.read_csv('NSE 500 Sector Weightages.csv')
                elif select_benchmark == 'ICICI Prudential Nifty 100 ETF Fund':
                    csv_file = "ICICIBenchmark.csv"
                    benchdata = yf.download("ICICINF100.NS", start = start_date, end = end_date)['Adj Close']
                    sectorbenchmark = pd.read_csv('ICICIBenchmark.csv')

                def build_bqm(alpha, _mu, _sigma, cardinality):
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
                tickers = constituents['Symbol'].to_list()
                use_local = False
                if use_local is False:
                        #end_date = end_date
                        #start_date = start_date
                        adj_close_df_1 = pd.DataFrame()
                        for ticker in tickers:
                            data = yf.download(ticker, start=start_date, end=end_date)
                            adj_close_df_1[ticker] = data['Adj Close']
                        adj_close_df_1.to_csv('benchmark.csv')
                
                #first_row_adj_close = adj_close_df.iloc[0]
                #total_budget = (first_row_adj_close * df['QUANTITY'].values).sum()
                #st.write(total_budget)

                def process_portfolio(init_holdings):
                    cfg.hpfilter_lamb = 6.25
                    cfg.q = 1.0  # risk-aversion factor
                    # classical
                    cfg.fmin = 0.01  # 0.001
                    cfg.fmax = 0.5  # 0.5
        
                    constituents = pd.read_csv(csv_file)
                    tickers = constituents['Symbol'].to_list()
                    data = pd.read_csv('benchmark.csv', parse_dates=['Date'])
                    sector_map = constituents.loc[constituents['Symbol'].isin(tickers)]
                    dates = data["Date"].to_numpy()
                    monthly_df = data.resample('3M', on='Date').last() # resample to every 3 months
                    month_end_dates = monthly_df.index
                    available_sectors, counts = np.unique(np.array(sector_map.sector.tolist()), return_counts=True)
                    #total_budget = 537787.2409820557 #Make the budget dynamic
                    global total_budget
                    total_budget = total_budget
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
                        mu = log_returns.mean().to_numpy() * working_days
                        sigma = log_returns.cov().to_numpy() * working_days
                        price = df.iloc[-1] # last day price

                        #Sell Idea
                        threshold = 4 # Sell all stocks for `threshold` companies
                        tickers_holding = np.array(list(init_holdings.keys())) # Names of the companies in initial holdings
                        indices = np.in1d(tickers, tickers_holding) # Indices of `tickers_holding` in the list of all companies `tickers`
                        argsort_indices = np.argsort(mu[indices]) # Obtain `mu` values at `indices`. Sort it.

                        sell_indices =  argsort_indices < threshold # indices of the least `threshold` companies (least in terms of mu value)
                        sell_tickers = tickers_holding[argsort_indices][sell_indices] # names of those companies

                        sectors = sector_map.loc[sector_map['Symbol'].isin(sell_tickers)]['sector'].tolist()
                        sectors = set(sectors) # remove duplicates

                        tickers_new = sector_map.loc[sector_map['sector'].isin(sectors)]['Symbol'].tolist()
                        tickers_new = np.intersect1d(np.array(tickers_new), np.array(tickers))
                        tickers_new = np.setdiff1d(np.array(tickers_new), np.array(sell_tickers))
          
                        keep_indices = np.in1d(np.array(tickers), tickers_new)
                        mu_new = mu[keep_indices]
                        sigma_new = sigma[keep_indices][:, keep_indices]

                        sales_revenue = 0.0
                        for tick in sell_tickers:
                            sales_revenue += init_holdings[tick] * price[tick]
                            init_holdings.pop(tick, None) # remove that company from holdings
                        bqm = build_bqm(cfg.q, mu_new, sigma_new, threshold)
                        sampler_sa = SimulatedAnnealingSampler()
                        result_sa = sampler_sa.sample(bqm, num_reads=5000)
                        selection = list(result_sa.first.sample.values())
                        selection = np.array(selection, dtype=bool)

                        tickers_selected = tickers_new[selection]
                        keep_indices = np.in1d(tickers_new, tickers_selected)
                        mu_selected = mu_new[keep_indices]
                        sigma_selected = sigma_new[keep_indices][:, keep_indices]

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

                        tickers = constituents['Symbol'].to_list()
                        row = [months[-1].strftime('%Y-%m-%d'), value_port/initial_budget] + \
                            [init_holdings.get(s, 0) for s in tickers] + \
                            [risk, returns, sr]
                        
                        opt_results_df.loc[i] = row.copy()
                        first_purchase = False
                    return opt_results_df
                
                #st.write(optimal_stocks_to_buy)
                #optimal_stocks_to_buy = {'BHARTIARTL.NS': 109.0, 'HDFCBANK.NS': 92.0, 'HINDUNILVR.NS': 92.0, 'ICICIBANK.NS': 104.0, 'INFY.NS': 86.0, 'ITC.NS': 112.0, 'LT.NS': 118.0, 'RELIANCE.NS': 107.0, 'SBIN.NS': 104.0, 'TCS.NS': 95.0, 'BAJFINANCE.NS':100.0, 'MARUTI.NS': 87.0, 'TITAN.NS':60.0}
                process_portfolio_amar = process_portfolio(optimal_stocks_to_buy)
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

                tickers = constituents['Symbol'].to_list()
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

                y_range = math.ceil(maximum_value / minimum_value )
                # Plotting
                fig_rebalancing = go.Figure()

                # Add trace for rebalanced portfolio value (divided by 10^7 to convert to crores)
                fig_rebalancing.add_trace(go.Scatter(
                    x=df2_cleaned['Date'],
                    y=df2_cleaned['Value'] / minimum_value,
                    mode='lines+markers',
                    name='Portfolio Value With Rebalancing',
                    line=dict(color='red'),
                    showlegend=True
                ))

                # Add trace for original portfolio value (divided by 10^7 to convert to crores)
                fig_rebalancing.add_trace(go.Scatter(
                    x=filtered_data['Date'],
                    y=filtered_data['portfolio value'] / minimum_value,
                    mode='lines+markers',
                    name='Portfolio Value Without Rebalancing',
                    line=dict(color='blue'),
                    showlegend=True
                ))

# Set layout and axis ranges
                fig_rebalancing.update_layout(
                    title="Rebalanced Portfolio Values Over Time",
                    xaxis_title='Date', 
                    yaxis_title=f"Portfolio Values scaled with Minimum portfolio value {round(minimum_value,2)}",
                    autosize=False, 
                    width=1000, 
                    height=600,
                    #yaxis_range=[0.15, 0.22]  # Adjust this range according to your data
                    yaxis_range = [0, y_range]
                    )

# Display the plot in Streamlit
                st.plotly_chart(fig_rebalancing)

    


                #st.write(adj_close_df)


                    
    else: 
        st.write("Please upload the Excel files to proceed.")
    
    
        
