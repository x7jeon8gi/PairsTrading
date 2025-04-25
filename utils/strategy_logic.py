import pandas as pd
import numpy as np
from typing import Tuple, Optional

def compute_spread_vectorized(group):
    """Vectorized helper function to compute spread within a group."""
    n = len(group)
    if n < 2:
        # Return Series with 0.0, preserving original index
        return pd.Series(0.0, index=group.index)

    # Sort the group by MOM1 value to identify high/low momentum firms
    sorted_group = group.sort_values()
    sorted_values = sorted_group.values
    sorted_indices = sorted_group.index

    mid_point = n // 2
    # Indices corresponding to the bottom and top halves based on sorted order
    bottom_indices = sorted_indices[:mid_point]
    top_indices = sorted_indices[n - mid_point:] # Indices of firms with higher momentum

    # Values of the bottom and top halves
    bottom_half_vals = sorted_values[:mid_point]
    top_half_vals = sorted_values[n - mid_point:]

    # Calculate the spread difference (High Momentum - Low Momentum)
    # This assumes pairing the lowest with highest, second lowest with second highest, etc.
    spread_diffs = top_half_vals - bottom_half_vals

    # Initialize Series to store results, indexed like the original group
    spreads = pd.Series(0.0, index=group.index)

    # Assign calculated spreads back to the corresponding firms
    # Firms in the top half (higher momentum) get a positive spread value
    spreads.loc[top_indices] = spread_diffs
    # Firms in the bottom half (lower momentum) get a negative spread value
    spreads.loc[bottom_indices] = -spread_diffs

    return spreads


def calculate_monthly_portfolio_log_return(
    current_cluster_data: pd.DataFrame,
    next_month_returns_series: pd.Series,
    threshold: float,
    stoploss: float, # Should be negative, e.g., -0.3
    current_prob_data: Optional[pd.DataFrame] = None,
    outlier_filter: Optional[float] = None,
) -> Tuple[float, pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Calculates the log return for one month based on clustering, momentum spread,
    and specified parameters. (Simplified version)
    """
    cluster_data = current_cluster_data.copy() # Work on a copy

    # --- Probability Filtering (Simplified) ---
    if current_prob_data is not None and outlier_filter is not None:
        prob_data = current_prob_data.copy()
        prob_data['max_prob'] = prob_data.max(axis=1)
        # Assume indices match or 'firms' column exists for merge
        combined_data = cluster_data.merge(prob_data[['max_prob']],
                                           left_index=True, # Simpler assumption
                                           right_index=True, # Simpler assumption
                                           how='left')
        combined_data['max_prob'].fillna(0, inplace=True) # Handle missing firms in prob_data
        quantile_threshold = combined_data['max_prob'].quantile(outlier_filter)
        combined_data['clusters'] = np.where(combined_data['max_prob'] > quantile_threshold,
                                             combined_data['clusters'], 0)
        cluster_data = combined_data[['clusters', 'MOM1']]
        cluster_data = cluster_data[cluster_data['clusters'] != 0]

    # --- Core Strategy Logic ---
    if cluster_data.empty:
        return 0.0, pd.DataFrame(), pd.DataFrame(), pd.Series(dtype=float)

    # Calculate spread using the vectorized function
    cluster_data['spread'] = cluster_data.groupby('clusters')['MOM1'].transform(compute_spread_vectorized)
    spread_std = cluster_data['spread'].replace(0, np.nan).std() # Exclude zeros for std calc
    if pd.isna(spread_std) or spread_std < 1e-8: spread_std = 1e-8 # Handle NaN/zero std

    # Determine positions based on spread sign and magnitude
    cluster_data['inPortfolio'] = cluster_data['spread'].abs() > (spread_std * threshold)
    # Spread > 0 means high momentum, Spread < 0 means low momentum in the pair context
    cluster_data['Long_or_Short'] = np.sign(-cluster_data['spread']) # High mom (spread>0) -> Short (-1), Low mom (spread<0) -> Long (1)

    long_firms = cluster_data[(cluster_data['Long_or_Short'] == 1) & cluster_data['inPortfolio']]
    short_firms = cluster_data[(cluster_data['Long_or_Short'] == -1) & cluster_data['inPortfolio']]

    # --- Calculate Portfolio Returns ---
    processed_next_returns = next_month_returns_series.copy()
    # Negate returns for short positions
    valid_short_indices = short_firms.index.intersection(processed_next_returns.index)
    processed_next_returns.loc[valid_short_indices] *= -1

    # Apply stop-loss and fill NaN
    stoploss_value = -np.abs(stoploss) # stoploss is expected negative
    condition_apply_stoploss = (processed_next_returns.notna()) & (processed_next_returns < stoploss_value)
    processed_next_returns = processed_next_returns.where(~condition_apply_stoploss, stoploss_value)
    processed_next_returns.fillna(-0.5, inplace=True)

    # --- Calculate Log Earning ---
    num_positions = len(long_firms) + len(short_firms)
    if num_positions == 0:
        log_earning = 0.0
    else:
        valid_long_indices = long_firms.index.intersection(processed_next_returns.index)
        valid_short_indices_for_sum = short_firms.index.intersection(processed_next_returns.index)
        long_sum = processed_next_returns.loc[valid_long_indices].sum()
        short_sum = processed_next_returns.loc[valid_short_indices_for_sum].sum() # Returns already negated
        log_earning = (long_sum + short_sum) / num_positions

    return log_earning, long_firms, short_firms, processed_next_returns
