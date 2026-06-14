# Music Analytics & Forecasting Pipeline (PySpark)

This project is a PySpark-based data engineering and machine learning pipeline designed to analyze user listening history from Last.fm datasets. It focuses on sessionizing time-series data to extract behavioral insights and perform future activity forecasting.

## Project Overview

- **Problem 2 (Analytics)**: Cleanses raw logs, sessionizes data based on a 20-minute gap threshold, and identifies the top 10 most played songs within the 50 longest sessions by track count.
- **Problem 3 (ML Forecasting)**: Extracts the user with the highest session frequency and utilizes a SARIMAX model (via `statsmodels`) to forecast their daily session counts for the next 90 days, including backtest validation.
- **Utilities**: A robust set of modular functions for sessionization, data cleansing, and metric aggregation.

## Project Structure

```text
.
├── Dockerfile              # Environment configuration for Spark & ML libs
├── Makefile                # CLI shortcuts for docker and spark-submit
├── docker-compose.yml      # Cluster orchestration (Master & Workers)
├── jobs/
│   ├── problem_2.py        # Analytics job (Top tracks/sessions)
│   ├── problem_3.py        # Forecasting job (User session prediction)
│   ├── utils/              # Shared logic modules
│   │   ├── constants.py    # Path and app constants
│   │   ├── metrics.py      # Aggregation and metric logic
│   │   └── utils.py        # Loading and sessionization logic
│   └── tests/              # Pytest suite
│       ├── run_tests.py    # Automated test runner with reporting
│       └── test_utils.py   # Unit tests for core functions
└── data/                   # Input/Output datasets (Volume-mapped from host machine)
```

## Key Assumptions

1.  **Sessionization Threshold**: A session is defined by a maximum 20-minute gap between track start times. Any longer gap indicates the start of a new session.
2.  **Data Integrity**: Listening logs missing critical identifiers (`userid`, `track-id`, or `track-name`) are considered noise and are filtered out during preprocessing.
3.  **User Context**: We only consider activity for users who exist in the `user-profile.tsv` dataset to ensure analysis is performed on validated user accounts.
4.  **Forecasting Model**: We assume weekly seasonality (7-day cycles) is a significant predictor for user listening habits, reflected in the `SARIMAX` seasonal order configuration.
5.  **Scalability of Aggregates**: While the raw logs are processed at scale via Spark, we assume that the aggregated daily session count for a single user is small enough to be handled by Pandas and Statsmodels in the container's memory for the forecasting step. 


## How to run the program

### Step 0: Download the Datasets
- Download from: http://ocelma.net/MusicRecommendationDataset/lastfm-1K.html
- Unzip the files and place in `/data/` directory
	- `userid-timestamp-artid-artname-traid-traname.tsv`
	- `userid-profile.tsv`

The scripts are executed via `spark-submit` on the `spark-master` container. We can run the below commands once Docker containers are setup (Refer to commands in previous step)

### Step 1: Top Songs Analytics Pipeline
```bash
make run_problem_2
```
*Output location:* `/opt/spark-data/output/top_songs_in_longest_sessions_by_track_count.csv`

### Step 2: Session Forecasting Pipeline
```bash
make run_problem_3
```
*Output location:* Check `/opt/spark-data/output/` for `forecast90d_user_most_sessions.csv` and the backtest evaluation report.

### Run any custom Spark job
This is useful for running any custom Spark job, but they must be located in the /jobs directory.
```bash
make run_job job=<JOB_NAME>.py
```


### Step 3: Running Unit Tests

The project includes a suite of tests using `pytest` to validate session and metric utility functions.

```bash
make run_tests
```
*After execution:* A detailed timestamped report is generated in the `tests/output/` directory (mapped to `/opt/spark-jobs/tests/output/` inside the container).



## Misc: Docker Environment Management

The project is fully containerized to ensure Spark, Python, and all dependencies (like `statsmodels` and `pandas`) are correctly configured.

### Step 0: Download the Datasets
- Download from: http://ocelma.net/MusicRecommendationDataset/lastfm-1K.html
- Unzip the files and place in `/data/` directory
	- `userid-timestamp-artid-artname-traid-traname.tsv`
	- `userid-profile.tsv`

### Miscellaneous: Helpful `make` Commands for Docker
To build the image and start the Spark cluster (Master and Workers) in the background:
```bash
make up
```

### Shutdown
To stop the services:
```bash
make down
```

### Restart
To restart the services:
```bash
make restart
```

### Nuke (Complete Reset)
To remove containers, networks, and images (useful if you want a clean slate):
```bash
make nuke
```

---