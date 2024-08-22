import os


def _extract_stock_name(data_file):
    stock_name_with_ext = os.path.basename(data_file)
    stock_name, _ = os.path.splitext(stock_name_with_ext)
    return stock_name
