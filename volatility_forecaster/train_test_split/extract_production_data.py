def extract_production_data(prod_size, lagged_df):
    prod_data = lagged_df.tail(int(len(lagged_df) * prod_size))
    return prod_data
