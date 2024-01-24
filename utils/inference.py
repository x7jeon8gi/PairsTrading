import h5py
from tqdm.auto import tqdm
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from openTSNE import TSNE
from sklearn.cluster import KMeans
import re
import os
from datetime import datetime
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

color_map = {
    0: '#1f77b4', 1: '#ff7f0e', 2: '#2ca02c', 3: '#d62728', 4: '#9467bd',
    5: '#8c564b', 6: '#e377c2', 7: '#7f7f7f', 8: '#bcbd22', 9: '#17becf',
    10: '#1a55FF', 11: '#E91E63', 12: '#9C27B0', 13: '#673AB7', 14: '#3F51B5',
    15: '#2196F3', 16: '#03A9F4', 17: '#00BCD4', 18: '#009688', 19: '#4CAF50',
    20: '#8BC34A', 21: '#CDDC39', 22: '#FFEB3B', 23: '#FFC107', 24: '#FF9800',
    25: '#FF5722', 26: '#795548', 27: '#9E9E9E', 28: '#607D8B', 29: '#000000'
}

# def get_hdf5_data(dataset: Dataset, ls: list):
#     """
#     :param dataset: Dataset object
#     :return: data in the dataset
#     """
#     pattern = r'[0-9]'
#     list_of_asset = [re.sub(pattern, '',os.path.basename(x).split('_')[2]) for x in ls]
#     dict_v = []
#     tmp = []
    
#     for idx, hdf5 in enumerate(tqdm(dataset.file_handle)):
#         data = {
#             'ticker': [hdf5['ticker'][i].decode('utf-8') for i in range(len(hdf5['ticker']))],
#             'date': [(hdf5['date'][i][0], hdf5['date'][i][-1]) for i in range(len(hdf5['ticker']))]
            
#         }
#         dict_v.append(pd.DataFrame(data))
        
#         repeated = np.repeat(list_of_asset[idx], len(hdf5['ticker']))
#         tmp.append(pd.DataFrame(repeated))
    
#     dict_df = pd.concat(dict_v, ignore_index=True)
#     dict_df['asset'] = pd.concat(tmp).values.ravel()
    
#     return dict_df


# def run_kmeans(latent_collector, n_cluster):
    
#     k_means = KMeans(n_clusters=n_cluster, random_state=random_state)
#     k_means = k_means.fit(latent_collector)
    
#     return k_means

def run_open_tsne(latent_collector, random_state, n_iter=500, metric='euclidean', n_jobs=32, verbose=True):
    """
    TSNE embedding을 계산합니다.
    openTSNE를 사용하므로 sklearn의 TSNE와 다른 인자를 사용합니다. (!pip install openTSNE)
    # https://opentsne.readthedocs.io/en/latest/api.html#opentsne.tsne.TSNE
    
    latent_collector: latent vector
    random_state: random seed
    n_iter: number of iteration
    metric: distance metric
    n_jobs: number of cpu
    verbose: verbose
    
    ------
    return: embedding
    """
    t_sne = TSNE(n_iter=n_iter, perplexity=500, metric=metric ,n_jobs=n_jobs, random_state=random_state, verbose=verbose)
    embedding = t_sne.fit(latent_collector)
    
    return embedding


# def get_latent(model, loader):
#     latent_collector = []
#     for idx, data in tqdm(loader):
#         data = data.to(config['model']['device'])
#         inputs = {"X": data}
#         inputs = model(inputs, training=False)
#         latent_collector.append(inputs["fcn_latent"].detach().cpu().numpy())

#     latent_collector = np.concatenate(latent_collector, axis=0)
    
#     return latent_collector

def get_index(df, latent, date):
    
    # get first date of date column
    date1 = df['date'].apply(lambda x: x.split(',')[0][1:]).astype(int)

    # after 2007 year index
    filtered_df = date1[date1 >= date]

    # get index
    filtered_idx = filtered_df.index

    # apply index to train_latent
    df_filtered = latent[filtered_idx]
    
    return df_filtered

import seaborn as sns
import matplotlib.pyplot as plt

def plot_ratio(df, figsize=(6,2), ratio_by='asset', x_label='Contrastive Class'):
    
    """
    df: DataFrame
    figsize: Size of the plot
    key_group: Group by this column
    x_label: Label for the x-axis
    
    Assuming you have a DataFrame called ratio_df with columns 'asset' and 'ticker'
    """

    def _get_ratio_df(df, key_group1, key_group2, level=0):
        count_df = pd.DataFrame(df.groupby([key_group1, key_group2], ).count()['ticker'])
        ratio_df = count_df.groupby(level=level ,group_keys=False).apply(lambda x: x / float(x.sum()))
        return ratio_df

    if ratio_by == 'asset':
        ratio_df = _get_ratio_df(df, 'asset', 'cluster', level=0)
        
    elif ratio_by == 'cluster':
        ratio_df = _get_ratio_df(df, 'asset', 'cluster', level=1)

    # Group and plot by each asset class
    for asset, data in ratio_df.groupby(ratio_by):
        fig, ax = plt.subplots(figsize=figsize)  # Create a new figure and axis for each asset class
        data.plot(kind='bar', ax=ax, label=asset, rot=0)
        
        # Annotate each bar with the ratio value
        for i, v in enumerate(data['ticker']):
            ax.text(i, v, f"{v:.2f}", fontsize=12, color='black', 
                    horizontalalignment='center', verticalalignment='bottom')
        
        # Customize the plot for each asset class
        ax.set_title(f"{asset} Ratio")
        ax.set_xlabel(x_label)
        ax.set_ylabel("Ratio")
        ax.legend(title=f'{asset}')#, loc='upper right')
        
        plt.tight_layout()
        plt.show()  # Show the individual plot for each asset class

def plot_radar_chart(test_df_with_input, day, label_column_name):
    """
    각 클러스터에 대한 Radar Chart를 그립니다.
    각 차트는 다음의 값을 나타냅니다.
    
    sharpe +
    volatility(양) -
    downside_deviation(양) -
    max_drawdown(음의 값)) +
    historical_var(음의 값)) +
    
    test_df_with_input: test 데이터프레임
    day: 20, 252 중 하나를 선택합니다.
    label_column_name: 실제 위험등급을 나타내는 컬럼의 이름을 입력합니다.
    """

    columns_to_plot = [f'sharpe_ratio_{day}', f'volatility_{day}', f'downside_deviation_{day}', f'max_drawdown_{day}', f'historical_var_{day}']
    for_plotting = test_df_with_input.copy()
    
    for_plotting[f'volatility_{day}'] *= -1
    for_plotting[f'downside_deviation_{day}'] *= -1
    
    for i in range(len(columns_to_plot)):
        scaler = MinMaxScaler()
        for_plotting[columns_to_plot[i]] = scaler.fit_transform(for_plotting[columns_to_plot[i]].values.reshape(-1, 1))

    grouped = for_plotting.groupby(label_column_name)[columns_to_plot]
    mean_values = grouped.mean()
    categories = mean_values.index.tolist()
    num_vars = len(columns_to_plot)

    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    
    for i, cat in enumerate(categories):
        cat_values = mean_values.loc[cat].tolist()
        cat_values += cat_values[:1]

        color_to_use = color_map.get(cat, 'black')  # Using the predefined color_map
        ax.plot(angles, cat_values, label=f'{cat}', color=color_to_use)
        ax.fill(angles, cat_values, color=color_to_use, alpha=0.25)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(columns_to_plot)

    ax.set_title('Mean Value Radar Chart for Each Label', size=20, color='black', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))

    plt.show()

# def plot_radar_chart_22(test_df_with_input, day, label_column_name):
#     """
#     각 클러스터에 대한 Radar Chart를 그립니다.
#     각 차트는 다음의 값을 나타냅니다.
    
#     sharpe +
#     volatility(양) -
#     downside_deviation(양) -
#     max_drawdown(음의 값)) +
#     historical_var(음의 값)) +
    
#     test_df_with_input: test 데이터프레임
#     day: 20, 252 중 하나를 선택합니다.
#     label_column_name: 실제 위험등급을 나타내는 컬럼의 이름을 입력합니다.
#     """
        
#     columns_to_plot = [f'sharpe_ratio_{day}', f'volatility_{day}', f'downside_deviation_{day}', f'max_drawdown_{day}', f'historical_var_{day}']
#     for_plotting = test_df_with_input.copy()
    
#     for_plotting[f'volatility_{day}'] *= -1
#     for_plotting[f'downside_deviation_{day}'] *= -1
    
#     for i in range(len(columns_to_plot)):
#         scaler = MinMaxScaler()
#         for_plotting[columns_to_plot[i]] = scaler.fit_transform(for_plotting[columns_to_plot[i]].values.reshape(-1, 1))

#     grouped = for_plotting.groupby(label_column_name)[columns_to_plot]
#     mean_values = grouped.mean()
#     categories = mean_values.index.tolist()
#     num_vars = len(columns_to_plot)

#     angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
#     angles += angles[:1]

#     fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    
#     for i, cat in enumerate(categories):
#         cat_values = mean_values.loc[cat].tolist()
#         cat_values += cat_values[:1]

#         color_to_use = color_map.get(cat, 'black')
#         ax.plot(angles, cat_values, label=f'{cat}', color=color_to_use)
#         ax.fill(angles, cat_values, alpha=0.25)

#     ax.set_xticks(angles[:-1])
#     ax.set_xticklabels(columns_to_plot)

#     ax.set_title('Mean Value Radar Chart for Each Label', size=20, color='black', pad=20)
#     ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))

#     plt.show()
    
def get_index2(df, date, up_down):

    if up_down == 'down':
        filtered_df = df[df['date'].apply(lambda x: x.split(',')[0][1:]).astype(int) < date]
    elif up_down =='up':
        filtered_df = df[df['date'].apply(lambda x: x.split(',')[0][1:]).astype(int) >= date]
    return filtered_df


def plot_specific_tickers(train_df, test_df, num_clusters, sort_with='max_drawdown_20'):
    """
    특정 티커에 대해서만 그래프를 그립니다.
    Train/Test Boundary를 나타내기 위해 2015년 1월 1일을 기준으로 세로 점선을 그립니다.
    실제 위험등급과 클러스터 번호의 변화를 볼 수 있습니다.
    
    train_df: train 데이터프레임
    test_df: test 데이터프레임
    sort_with: 'sharpe_ratio_20', 'volatility_20', 'downside_deviation_20', 'max_drawdown_20', 'historical_var_20' 중 하나를 선택합니다.
    """
    tickers=['대한민국 - KOSPI', 'KOSEF 국고채3년', '주요상품선물_금(선물)', '삼성전자', '현대글로비스', '금호타이어', 'Apple', 'S&P 500']
    eng_tickers = ['Korea - KOSPI', 'KOSEF Treasury Bond 3yr', 'Gold Futures', 'Samsung Electronics', 'Hyundai Glovis', 'Kumho Tire', 'Apple', 'S&P 500']

    mean_values = test_df.groupby('cluster').mean().sort_values(by=sort_with)

    for eng, ticker in zip(eng_tickers, tickers):

        train_df_filtered = train_df[train_df['ticker']==f'{ticker}'].sort_values(by=['date'])
        test_df_filtered = test_df[test_df['ticker']==f'{ticker}'].sort_values(by=['date'])
        
        date_train = train_df_filtered['date'].values
        date_test = test_df_filtered['date'].values
        date_all = np.concatenate([date_train, date_test])
        
        real_value_train = train_df_filtered['cluster']
        real_value_test = test_df_filtered['cluster']
        real_value_all = np.concatenate([real_value_train, real_value_test])
        
        labels_ts = mean_values.index.tolist()[::-1]
        mapping = {ticker: idx for idx, ticker in enumerate(labels_ts)}
        
        new_ts_all = np.array([mapping[ticker] for ticker in real_value_all])
        
        plt.figure(figsize=(10,3))
        plt.plot(date_all, new_ts_all, label='Risk Level')
        plt.plot(date_all, real_value_all, label='Cluster Number')
        
        plt.ylim(0, num_clusters)  # y축 값을 0부터 num_clusters까지로 설정
        
        plt.axvline(x=mdates.date2num(pd.Timestamp('2015-01-01')), color='r', linestyle='--', linewidth=1, label='Train/Test Boundary')
        
        plt.legend(loc='upper left')
        plt.xlabel('Date')
        plt.ylabel('Cluster Value')
        plt.title(f'Ticker: {eng}')
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.show()


# def calculate_portfolio_returns(df):
#     """
#     각 클러스터에 대한 포트폴리오 수익률을 계산합니다.
#     df: return, date, cluster, and ticker가 포함된 DataFrame
#     """
    
#     # 클러스터 및 날짜별 평균 수익률 계산
#     grouped_returns = df.groupby(['cluster', 'date'])['return'].mean().unstack()

#     # 20일 간격으로 선택 및 평균 계산
#     portfolio_dates = grouped_returns.columns[::20].tolist()
#     if grouped_returns.columns[-1] not in portfolio_dates:
#         portfolio_dates.append(grouped_returns.columns[-1])
    
#     # 20일 간격과 해당 날짜 사이의 평균을 계산
#     avg_returns = [pd.Series([0])] # 시작일은 0으로 설정
#     for start, end in zip(portfolio_dates[:-1], portfolio_dates[1:]):
#         avg_returns.append(grouped_returns.loc[:, start:end].mean(axis=1))
#     avg_returns_df = pd.concat(avg_returns, axis=1)
#     avg_returns_df.fillna(0, inplace=True)
    
#     avg_returns_df.columns = portfolio_dates[0:]

#     return avg_returns_df.transpose().rename(lambda x: f'Cluster_{x}', axis=1)


def calculate_portfolio_returns(df, outlier_threshold=4):
    """
    포트폴리오 수익률을 계산합니다.
    ----
    df: return, date, cluster, and ticker가 포함된 DataFrame
    outlier_threshold: 수익률의 이상치를 제거합니다.
    ----
    return: combined, cluster_avg_return, cluster_cum_return, df
    combined: 날짜별, 클러스터별 티커 수와 수익률의 합계를 포함하는 DataFrame
    cluster_avg_return: 클러스터별 평균 수익률
    cluster_cum_return: 클러스터별 누적 수익률
    df: outlier_threshold를 적용한 DataFrame
    """
    # ticker별로 정렬
    df = df.sort_values(by=["ticker", "date"])

    # ticker별로 date의 차이 계산
    df['date_diff'] = df.groupby('ticker')['date'].diff().dt.days

    # date 차이가 50일 이상인 지점마다 ticker의 이름을 변경
    df['ticker_change'] = (df['date_diff'] >= 60).cumsum()
    df['modified_ticker'] = df['ticker'] + "_" + df['ticker_change'].astype(str)
    
    # ticker별 수익률 계산
    df['20d_return'] = df.groupby('modified_ticker')['price'].pct_change()
    df['20d_return'] = df.groupby('modified_ticker')['20d_return'].shift(-1)
    
    # df['20d_return'] = df.groupby('ticker')['price'].pct_change()
    # df['20d_return'] = df.groupby('ticker')['20d_return'].shift(-1)
    
    df['20d_return'] = df['20d_return'].fillna(0)
    df = df[df['20d_return'] < outlier_threshold] # todo: data issue
    
    # 날짜별, 클러스터별 티커 수 계산
    ticker_counts = df.groupby(['date', 'cluster']).size().rename('ticker_count')
    
    # 날짜별, 클러스터별 수익률의 합계 계산
    total_returns = df.groupby(['date', 'cluster'])['20d_return'].sum().rename('total_return')
    
    # DataFrame 병합
    combined = pd.concat([ticker_counts, total_returns], axis=1)
    combined['avg_return'] = combined['total_return'] / combined['ticker_count']
    
    # 클러스터별 평균 수익률 Pivot
    cluster_avg_return = combined.pivot_table(values='avg_return', index='date', columns='cluster')
    
    # 클러스터별 누적 수익률 계산
    cluster_cum_return = (cluster_avg_return + 1).cumprod()
    
    return combined ,cluster_avg_return, cluster_cum_return, df

# def calculate_portfolio_returns(df):
#     # ticker별로 정렬
#     grouped = df.sort_values(by=["ticker", "date"])
    
#     # ticker별 수익률 계산
#     grouped['20d_return'] = grouped.groupby('ticker')['price'].pct_change()
#     grouped['20d_return'] = grouped['20d_return'].fillna(0)
#     grouped = grouped[grouped['20d_return'] < 1] # todo: data issue

#     # 날짜별, 클러스터별 티커 수 계산
#     ticker_counts = grouped.groupby(['date', 'cluster']).size().rename('ticker_count')
    
#     # 날짜별, 클러스터별 수익률의 합계 계산
#     total_returns = grouped.groupby(['date', 'cluster'])['20d_return'].sum().rename('total_return')
    
#     # DataFrame 병합
#     combined = pd.concat([ticker_counts, total_returns], axis=1)
#     combined['avg_return'] = combined['total_return'] / combined['ticker_count']
    
#     # 클러스터별 평균 수익률 Pivot
#     cluster_avg_return = combined.pivot_table(values='avg_return', index='date', columns='cluster')
    
#     # 클러스터별 누적 수익률 계산
#     cluster_cum_return = (cluster_avg_return + 1).cumprod()
    
#     return combined ,cluster_avg_return, cluster_cum_return


def plot_portfolio_returns(portfolio_returns, ylim=None):
    """
    미리 정의된 컬러 맵을 사용하여 포트폴리오 수익률을 그립니다.
    
    - portfolio_returns: pandas 데이터프레임
    """

    # Set the style
    plt.style.use('ggplot')

    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(15, 5))

    # Plot the data using colors from the color map
    if isinstance(portfolio_returns, pd.DataFrame):
        for col, color_key in zip(portfolio_returns.columns, color_map):
            ax.plot(portfolio_returns.index, portfolio_returns[col], color=color_map[color_key], label=col)
    elif isinstance(portfolio_returns, pd.Series):
        ax.plot(portfolio_returns.index, portfolio_returns, color=color_map[0])
    else:
        raise ValueError("Input should be a pandas DataFrame or Series.")

    if ylim is not None:
        ax.set_ylim(ylim)
    # Add a title
    ax.set_title('Portfolio Returns')

    # Add x and y axis labels
    ax.set_xlabel('Date')
    ax.set_ylabel('Return')

    # Add a legend if multiple columns
    if isinstance(portfolio_returns, pd.DataFrame) and len(portfolio_returns.columns) > 1:
        ax.legend()

    # Show the plot
    plt.tight_layout()
    plt.show()
    

def plot_cumulative_returns(portfolio_returns, ylim=None):
    """
    미리 정의된 컬러 맵을 사용하여 포트폴리오의 누적 수익률을 그립니다
    
    portfolio_returns: pandas 데이터프레임
    """
    
    # Calculate cumulative returns
    cumulative_returns = (portfolio_returns + 1).cumprod()
    
    # Set the style
    plt.style.use('ggplot')

    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(15, 5))

    # Plot the data using colors from the color map
    if isinstance(cumulative_returns, pd.DataFrame):
        for col, color_key in zip(cumulative_returns.columns, color_map):
            ax.plot(cumulative_returns.index, cumulative_returns[col], color=color_map[color_key], label=col)
    elif isinstance(cumulative_returns, pd.Series):
        ax.plot(cumulative_returns.index, cumulative_returns, color=color_map[0])
    else:
        raise ValueError("Input should be a pandas DataFrame or Series.")

    # Add a title
    ax.set_title('Cumulative Portfolio Returns')
    if ylim is not None:
        ax.set_ylim(ylim)
    # Add x and y axis labels
    ax.set_xlabel('Date')
    ax.set_ylabel('Cumulative Return')

    # Add a legend if multiple columns
    if isinstance(cumulative_returns, pd.DataFrame) and len(cumulative_returns.columns) > 1:
        ax.legend()

    # Show the plot
    plt.tight_layout()
    plt.show()
