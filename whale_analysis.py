#!/usr/bin/env python
# coding: utf-8

#  #  A Whale off the Port(folio)
#  ---
# 
#  In this assignment, you'll get to use what you've learned this week to evaluate the performance among various algorithmic, hedge, and mutual fund portfolios and compare them against the S&P TSX 60 Index.

# In[67]:


# Initial imports
import pandas as pd
import numpy as np
import datetime as dt
import seaborn as sb
import matplotlib.pyplot as plt
from pathlib import Path


sb.set_theme(context='talk', style='darkgrid', palette='Paired' ,font='sans-serif', font_scale=1, color_codes=True, rc=None)
# %matplotlib inline


# In[136]:


#Functions to read and clean data.

#This function reads a CSV file into a DataFrame and if dateIndex is True then it sets Date column to te the index
def read_data(file, dateIndex):
    df = pd.read_csv(file, index_col="Date", parse_dates=True, infer_datetime_format=True) if dateIndex else pd.read_csv(file)
    
    return df

#This function cleans the dataframe passed by clearing all the null values and stripping whitespace from column names
def clean_data(df):
    df = df.dropna()
    df = df.rename(columns=lambda x: x.strip()) #Lambda function to strip whitespace from column names
    return df
        


# # Data Cleaning
# 
# In this section, you will need to read the CSV files into DataFrames and perform any necessary data cleaning steps. After cleaning, combine all DataFrames into a single DataFrame.
# 
# Files:
# 
# * `whale_returns.csv`: Contains returns of some famous "whale" investors' portfolios.
# 
# * `algo_returns.csv`: Contains returns from the in-house trading algorithms from Harold's company.
# 
# * `sp_tsx_history.csv`: Contains historical closing prices of the S&P TSX 60 Index.

# ## Whale Returns
# 
# Read the Whale Portfolio daily returns and clean the data.

# In[69]:


whale_file = Path('./Resources/whale_returns.csv')


# In[70]:


# Reading whale returns
whale_df = read_data(whale_file, True)


# In[71]:


# Count nulls
whale_df.isnull().sum()


# In[72]:


# Drop nulls
whale_df = clean_data(whale_df)


# ## Algorithmic Daily Returns
# 
# Read the algorithmic daily returns and clean the data.

# In[73]:


algo_file = Path('./Resources/algo_returns.csv')


# In[74]:


# Reading algorithmic returns
algo_df = read_data(algo_file, True)


# In[75]:


# Count nulls
algo_df.isnull().sum()


# In[76]:


# Drop nulls
algo_df = clean_data(algo_df)


# ## S&P TSX 60 Returns
# 
# Read the S&P TSX 60 historic closing prices and create a new daily returns DataFrame from the data. 

# In[77]:


sp_file = Path('./Resources/sp_tsx_history.csv')


# In[78]:


# Reading S&P TSX 60 Closing Prices
sp_tsx_df = read_data(sp_file, True)
sp_tsx_df.head()


# In[79]:


# Check Data Types
sp_tsx_df.info()


# In[80]:


# Fix Data Types
sp_tsx_df['Close'] = sp_tsx_df['Close'].str.replace('$','')
sp_tsx_df['Close'] = sp_tsx_df['Close'].apply(lambda x : float(x.replace(',',''))) #Replacing coma with blank and converting string to float using lambda
sp_tsx_df.info()


# In[81]:


# Calculate Daily Returns
sp_tsx_df.head()
sp_tsx_dr = sp_tsx_df.pct_change()
sp_tsx_dr.head(20)


# In[82]:


# Drop nulls
sp_tsx_dr.isnull().sum()
sp_tsx_dr = clean_data(sp_tsx_dr)


# In[83]:


# Rename `Close` Column to be specific to this portfolio.
sp_tsx_dr.columns = ['SP TXS DR']


# ## Combine Whale, Algorithmic, and S&P TSX 60 Returns

# In[84]:


# Join Whale Returns, Algorithmic Returns, and the S&P TSX 60 Returns into a single DataFrame with columns for each portfolio's returns.
combined_dr = pd.concat([whale_df, algo_df, sp_tsx_dr], axis='columns', join='inner')
combined_dr.head()


# ---

# # Conduct Quantitative Analysis
# 
# In this section, you will calculate and visualize performance and risk metrics for the portfolios.

# ## Performance Anlysis
# 
# #### Calculate and Plot the daily returns.

# In[139]:


# Plot daily returns of all portfolios

combined_dr_plot = combined_dr['SOROS FUND MANAGEMENT LLC'].plot()
combined_dr_plot = combined_dr['PAULSON & CO.INC.'].plot(ax=combined_dr_plot)
combined_dr_plot = combined_dr['TIGER GLOBAL MANAGEMENT LLC'].plot(ax=combined_dr_plot)
combined_dr_plot = combined_dr['BERKSHIRE HATHAWAY INC'].plot(ax=combined_dr_plot)
combined_dr_plot = combined_dr['Algo 1'].plot(ax=combined_dr_plot)
combined_dr_plot = combined_dr['Algo 2'].plot(ax=combined_dr_plot)
combined_dr_plot = combined_dr['SP TXS DR'].plot(figsize=(20,10), title= 'Daily Returns of all portfolios')

combined_dr_plot.legend(['SOROS FUND', 'PAULSON INC', 'TIGER GLOBAL', 'BERKSHIRE INC', 'Algo 1', 'Algo 2', 'SP TXS'])


# In[141]:


# this logic separately plots each Daily return of each fund paired with Daily return of SP TXS. This allows us to look at each fund individually to measure performance in comparison to SP TSX

#This function extract values for a fund name and SP TSX and creates a dataframe with an additional column Fund Performance
def get_melted_DF(df, fund_name):
    df = pd.melt(df.reset_index(), id_vars=['Date'], var_name='Fund Name', value_name='Daily Return', value_vars=[fund_name, 'SP TXS DR'])
    df['Fund Performance'] = fund_name
    return df

# Merges all newly created data frames
def merge_DF(df_tomerge, df):
    return df.merge(df_tomerge, how='outer', on=['Date','Fund Name', 'Daily Return', 'Fund Performance'])

paulson_melted_df = get_melted_DF(combined_dr, 'PAULSON & CO.INC.')
soros_melted_df = get_melted_DF(combined_dr, 'SOROS FUND MANAGEMENT LLC')
tiger_melted_df = get_melted_DF(combined_dr, 'TIGER GLOBAL MANAGEMENT LLC')
berkshire_melted_df = get_melted_DF(combined_dr, 'BERKSHIRE HATHAWAY INC')
algo_1_melted_df = get_melted_DF(combined_dr, 'Algo 1')
algo_2_melted_df = get_melted_DF(combined_dr, 'Algo 2')

combined_melted_df = merge_DF(paulson_melted_df, soros_melted_df)
combined_melted_df = merge_DF(tiger_melted_df, combined_melted_df)
combined_melted_df = merge_DF(berkshire_melted_df, combined_melted_df)
combined_melted_df = merge_DF(algo_1_melted_df, combined_melted_df)
combined_melted_df = merge_DF(algo_2_melted_df, combined_melted_df)

# Subplots each fund comparison with SP TSX along Fund performance and Fund name
graphs = sb.FacetGrid(combined_melted_df, hue='Fund Name', row='Fund Performance', height=5, aspect=3)
graphs.map(sb.lineplot, 'Date', 'Daily Return', alpha=.7)
graphs.add_legend()


# #### Calculate and Plot cumulative returns.

# In[142]:


# Calculate cumulative returns of all portfolios

cumulative_returns = (1 + combined_dr).cumprod() - 1
cumulative_returns

# Plot cumulative returns
plot_cr = cumulative_returns.plot(figsize=(20,10), title="Cumulative Returns")
plot_cr.set_ylabel('Cumulative Returns')
plot_cr.set_xlabel('Date')


# ---

# ## Risk Analysis
# 
# Determine the _risk_ of each portfolio:
# 
# 1. Create a box plot for each portfolio. 
# 2. Calculate the standard deviation for all portfolios.
# 4. Determine which portfolios are riskier than the S&P TSX 60.
# 5. Calculate the Annualized Standard Deviation.

# ### Create a box plot for each portfolio
# 

# In[144]:


# Box plot to visually show risk
f, ax = plt.subplots(figsize=(35, 17.5))
box_plot = sb.boxplot(data=combined_dr, fliersize=10)


# ### Calculate Standard Deviations

# In[89]:


# Calculate the daily standard deviations of all portfolios
combined_std = combined_dr.std().sort_values(ascending=False)
combined_std


# ### Determine which portfolios are riskier than the S&P TSX 60

# In[90]:


# Calculate  the daily standard deviation of S&P TSX 60
sp_std = combined_dr['SP TXS DR'].std()
sp_std
# Determine which portfolios are riskier than the S&P TSX 60


# ### Calculate the Annualized Standard Deviation

# In[91]:


# Calculate the annualized standard deviation (252 trading days)
annulised_combined_std = combined_std * np.sqrt(252)
annulised_combined_std


# ---

# ## Rolling Statistics
# 
# Risk changes over time. Analyze the rolling statistics for Risk and Beta. 
# 
# 1. Calculate and plot the rolling standard deviation for all portfolios using a 21-day window.
# 2. Calculate the correlation between each stock to determine which portfolios may mimick the S&P TSX 60.
# 3. Choose one portfolio, then calculate and plot the 60-day rolling beta for it and the S&P TSX 60.

# ### Calculate and plot rolling `std` for all portfolios with 21-day window

# In[92]:


# Calculate the rolling standard deviation for all portfolios using a 21-day window
combined_std_21 = combined_dr.rolling(window=21).std()

# Plot the rolling standard deviation
plot_std_21 = combined_std_21.plot(figsize=(20,10), title="21 Day Rolling Standard Deviation")
plot_std_21.set_xlabel('Date')
plot_std_21.set_ylabel('21 Days Rolling Standard Deviation')


# ### Calculate and plot the correlation

# In[93]:


# Calculate the correlation
combined_corr = combined_dr.corr()
# Display de correlation matrix
combined_corr.style.background_gradient(cmap='coolwarm')


# In[128]:


f, ax = plt.subplots(figsize=(11, 9))
sb.heatmap(combined_corr, annot=True)


# ### Calculate and Plot Beta for a chosen portfolio and the S&P 60 TSX

# In[94]:


# Calculate covariance of a single portfolio
covar_berkshire_60 = combined_dr['BERKSHIRE HATHAWAY INC'].rolling(window=60).cov(combined_dr['SP TXS DR'])

# Calculate variance of S&P TSX
variance_sp_60 = combined_dr['SP TXS DR'].rolling(window=60).var()

# Computing beta
beta_berkshire_60 = covar_berkshire_60 / variance_sp_60

# Plot beta trend
plot_beta_60 = beta_berkshire_60.plot(figsize=(15,10), title="Berkshire Beta Trends")
plot_beta_60.set_ylabel('Beta')


# ## Rolling Statistics Challenge: Exponentially Weighted Average 
# 
# An alternative way to calculate a rolling window is to take the exponentially weighted moving average. This is like a moving window average, but it assigns greater importance to more recent observations. Try calculating the [`ewm`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.ewm.html) with a 21-day half-life.

# In[95]:


# Use `ewm` to calculate the rolling window
combined_ewm_21 = combined_dr.ewm(halflife=21).mean()
combined_ewm_21


# ---

# # Sharpe Ratios
# In reality, investment managers and thier institutional investors look at the ratio of return-to-risk, and not just returns alone. After all, if you could invest in one of two portfolios, and each offered the same 10% return, yet one offered lower risk, you'd take that one, right?
# 
# ### Using the daily returns, calculate and visualize the Sharpe ratios using a bar plot

# In[96]:


# Annualized Sharpe Ratios
combine_spr = (combined_dr.mean() * 252) /(combined_std * np.sqrt(252))
combine_spr


# In[97]:


# Visualize the sharpe ratios as a bar plot
combine_spr.plot.bar(title="Sharpe Ratios")


# ### Determine whether the algorithmic strategies outperform both the market (S&P TSX 60) and the whales portfolios.
# 
# Write your answer here!

# Based on the graph above, it can be concluded that algorithmic strateties have performed better by taking more risks as compared to S&P TXS 60 and other whale portfolios.

# # Create Custom Portfolio
# 
# In this section, you will build your own portfolio of stocks, calculate the returns, and compare the results to the Whale Portfolios and the S&P TSX 60. 
# 
# 1. Choose 3-5 custom stocks with at last 1 year's worth of historic prices and create a DataFrame of the closing prices and dates for each stock.
# 2. Calculate the weighted returns for the portfolio assuming an equal number of shares for each stock.
# 3. Join your portfolio returns to the DataFrame that contains all of the portfolio returns.
# 4. Re-run the performance and risk analysis with your portfolio to see how it compares to the others.
# 5. Include correlation analysis to determine which stocks (if any) are correlated.

# ## Choose 3-5 custom stocks with at last 1 year's worth of historic prices and create a DataFrame of the closing prices and dates for each stock.
# 
# For this demo solution, we fetch data from three companies listes in the S&P TSX 60 index.
# 
# * `SHOP` - [Shopify Inc](https://en.wikipedia.org/wiki/Shopify)
# 
# * `OTEX` - [Open Text Corporation](https://en.wikipedia.org/wiki/OpenText)
# 
# * `L` - [Loblaw Companies Limited](https://en.wikipedia.org/wiki/Loblaw_Companies)

# In[98]:


# Reading data from 1st stock
alphabet_file = Path('./Resources/alphabet_historical.csv')
alphabet_df = read_data(alphabet_file, True)
alphabet_df.columns = ['Alphabet_Close']
alphabet_df


# In[99]:


# Reading data from 2nd stock
tesla_file = Path('./Resources/tesla_historical.csv')
tesla_df = read_data(tesla_file, True)
tesla_df.columns = [ 'Tesla_Close']
tesla_df


# In[100]:


# Reading data from 3rd stock
apple_file = Path('./Resources/apple_historical.csv')
apple_df = read_data(apple_file, True)
apple_df.columns = [ 'Apple_Close']
apple_df


# In[101]:


# Combine all stocks in a single DataFrame
custom_combined_df = pd.concat([alphabet_df, tesla_df, apple_df], axis='columns', join='inner')
custom_combined_df


# In[102]:


# Reset Date index
custom_combined_df = custom_combined_df.reset_index()
custom_combined_df


# In[103]:


# Reorganize portfolio data by having a column per symbol
custom_combined_df = pd.pivot_table(custom_combined_df, index=['Date'])
custom_combined_df


# In[104]:


# Calculate daily returns
custom_combined_dr = custom_combined_df.pct_change()
custom_combined_dr
# Drop NAs
custom_combined_dr = clean_data(custom_combined_dr)
custom_combined_dr.columns = ['Alphabet', 'Tesla', 'Apple']
# Display sample data
custom_combined_dr.head(5)


# ## Calculate the weighted returns for the portfolio assuming an equal number of shares for each stock

# In[105]:


# Set weights
weights = [1/3, 1/3, 1/3]

# Calculate portfolio return
custom_weighted_returns = custom_combined_dr.dot(weights)
# Display sample data
custom_weighted_returns.head(5)


# ## Join your portfolio returns to the DataFrame that contains all of the portfolio returns

# In[106]:


# Join your returns DataFrame to the original returns DataFrame
all_portfolio_dr = pd.concat([combined_dr, custom_combined_dr], axis='columns', join='inner')
all_portfolio_dr


# In[145]:


# Only compare dates where return data exists for all the stocks (drop NaNs)
# There is no need to drop nulls because inner joining would only select the dates where there is data across all columns


# ## Re-run the risk analysis with your portfolio to see how it compares to the others

# ### Calculate the Annualized Standard Deviation

# In[108]:


# Calculate the annualized `std`
all_portfolio_std = all_portfolio_dr.std().sort_values(ascending=False)
all_portfolio_std_annulised = all_portfolio_std * np.sqrt(252)
all_portfolio_std_annulised


# ### Calculate and plot rolling `std` with 21-day window

# In[109]:


# Calculate rolling standard deviation
all_portfolio_std_21 = all_portfolio_dr.rolling(window=21).std()

# Plot the rolling standard deviation
plot_std_21 = all_portfolio_std_21.plot(figsize=(20,10), title="21 Day Rolling Standard Deviation for All Portfolios")
plot_std_21.set_xlabel('Date')
plot_std_21.set_ylabel('21 Days Rolling Standard Deviation')


# ### Calculate and plot the correlation

# In[110]:


# Calculate and plot the correlation
all_portfolio_corr = all_portfolio_dr.corr()

all_portfolio_corr.style.background_gradient(cmap='coolwarm')


# In[147]:


f, ax = plt.subplots(figsize=(11, 9))
sb.heatmap(all_portfolio_corr, annot=True)


# ### Calculate and Plot the 60-day Rolling Beta for Your Portfolio compared to the S&P 60 TSX

# In[111]:


# Calculate and plot Beta
covar_alphabet_60 = all_portfolio_dr['Alphabet'].rolling(window=60).cov(all_portfolio_dr['SP TXS DR'])
covar_apple_60 = all_portfolio_dr['Apple'].cov(all_portfolio_dr['SP TXS DR'])
covar_tesla_60 = all_portfolio_dr['Tesla'].cov(all_portfolio_dr['SP TXS DR'])

# Calculate variance of S&P TSX
variance_sp_60 = all_portfolio_dr['SP TXS DR'].rolling(window=60).var()

# Computing beta
beta_alphabet_60 = covar_alphabet_60 / variance_sp_60
beta_apple_60 = covar_apple_60 / variance_sp_60
beta_tesla_60 = covar_tesla_60 / variance_sp_60


# Plot beta trend
plot_beta_60 = beta_alphabet_60.plot()
plot_beta_60 = beta_apple_60.plot(ax=plot_beta_60)
plot_beta_60 = beta_tesla_60.plot(ax = plot_beta_60, figsize=(15,10), title="Custom Portfolio Beta Trends")
plot_beta_60.set_ylabel('Beta')
plot_beta_60.legend(['Alphabet Beta', 'Apple Beta', 'Tesla Beta'])


# ### Using the daily returns, calculate and visualize the Sharpe ratios using a bar plot

# In[112]:


# Calculate Annualzied Sharpe Ratios
all_portfolio_spr = (all_portfolio_dr.mean() * 252) /(all_portfolio_dr.std() * np.sqrt(252))
all_portfolio_spr


# In[114]:


# Visualize the sharpe ratios as a bar plot
all_portfolio_spr.plot.bar(title="Sharpe Ratios")


# ### How does your portfolio do?
# 
# Write your answer here!
# 
# Based on the above graph, between 1st January 2017 and 31st December 2018, my portfolio comprising of Alphabet INC, Tesla INC and Apple INC have performed on par with Algorithmic portfolio comprising of Algo 1 and Algo 2 funds. But the portfoliohas fared better compared to S&P TSX market and other whale funds. It has provided good returns for greater risks taken.

# 
