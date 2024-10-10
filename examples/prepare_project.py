"preprocessing the data for the project"

from volatility_forecaster.preprocessing.compute_log_yield import compute_log_yield
from volatility_forecaster.preprocessing.compute_rolling_std import compute_rolling_std
from volatility_forecaster.preprocessing.copy_data_from_intermediate_to_processed import (
    copy_data_from_intermediate_to_processed,
)
from volatility_forecaster.preprocessing.copy_data_from_raw_to_intermediate import (
    copy_data_from_raw_to_intermetidate,
)
from volatility_forecaster.pull_data.download_full_data_from_yahoo import (
    download_full_data_from_yahoo,
)
from volatility_forecaster.start_project.create_project import create_project

# 1. prepare the directories for the project
PROJECT_NAME = "USA"

create_project(project_name=PROJECT_NAME)

# 2. download data from yahoo using nemotecnics of the stocks
stocks_list = ["AAPL", "MSFT", "GOOGL"]
start_date = "2018-01-01"
end_date = "2024-05-31"


download_full_data_from_yahoo(
    stocks_list=stocks_list,
    start_date=start_date,
    end_date=end_date,
    project_name=PROJECT_NAME,
)

# 3. remove N/A's
copy_data_from_raw_to_intermetidate(project_name=PROJECT_NAME)

# 4. format the data for the project
copy_data_from_intermediate_to_processed(project_name=PROJECT_NAME)

# 5. compute yields
compute_log_yield(project_name=PROJECT_NAME)

# 6. compute rolling std
compute_rolling_std(project_name=PROJECT_NAME, rolling_window=5)
