"""
Problem: PySpark Analytics Pipeline
Author: Anando Zaman
Date: June 13, 2026

Problem/Task:
---------------------
Given the below datasets, write a PySpark job to answer the following:
    - What are the top 10 songs played in the top 50 longest sessions by tracks count?

In this assignment, a user "session" consists of one or more songs played by a given
user, where each song is started within 20 minutes of the previous song's start time.

Datasets:
--------------------
DOWNLOAD: http://ocelma.net/MusicRecommendationDataset/lastfm-1K.html

 - Dataset A: userid-timestamp-artid-artname-traid-traname.tsv
    - SCHEMA: userid \t timestamp \t musicbrainz-artist-id \t artist-name \t musicbrainz-track-id \t track-name
- Dataset B: userid-profile.tsv:
    - SCHEMA: userid \t gender ('m'|'f'|empty) \t age (int|empty) \t country (str|empty) \t signup (date|empty)



Solution:
---------------------
Steps as follows:
1. Load the two datasets into PySpark DataFrames.
2. Clean the datasets:
    - Filter out null IDs (eg; userid, musicbrainz-track-id, track-name)
    - Filter out users from dataset A that do not exist in dataset B.
        - dataset A is small, thus perform broadcast join for performance gains when joining with dataset B
3. Sessionize the data - define a global_session_key
    - Use the timestamp column to determine previous timestamps for each user and determine the time gap between current and previous timestamp. 
        - If the time gap is greater than 20 mins (or if first track), flag as new session.
    - Generate a global_session_key by concatenating the userid and session_id
4. Determine top 50 longest sessions by track count (i.e. number of tracks played in the session)
5. Determine the top 10 tracks played from the previous step data.
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql import DataFrame, types as T
from pyspark.sql.types import StructType, StructField, StringType
from utils.metrics import get_M_longest_sessions_by_track_count, get_top_N_songs_by_count
from utils.utils import load_dataset, generate_session_info, preprocess_music_data
from utils.constants import SPARK_MASTER, MUSIC_ANALYTICS_PIPELINE, MUSIC_USER_TIMESTAMPS_DATASET_DIR, USER_PROFILES_DATASET_DIR, DATASET_OUTPUT_DIRECTORY

if __name__ == "__main__":
    # Initialize Spark session
    spark_session = SparkSession.builder.master(SPARK_MASTER).appName(MUSIC_ANALYTICS_PIPELINE).getOrCreate()

    music_user_timestamps_schema = StructType([
        StructField("userid", StringType(), True),
        StructField("timestamp", T.TimestampType(), True), # Use TimestampType for proper sessionization
        StructField("musicbrainz-artist-id", StringType(), True),
        StructField("artist-name", StringType(), True),
        StructField("musicbrainz-track-id", StringType(), True),
        StructField("track-name", StringType(), True)
    ])
    
    # STEP 1: Load datasets
    music_user_timestamps_df = load_dataset(spark_session=spark_session, data_path=MUSIC_USER_TIMESTAMPS_DATASET_DIR, schema=music_user_timestamps_schema)
    user_profiles_df = load_dataset(spark_session=spark_session, data_path=USER_PROFILES_DATASET_DIR).withColumnRenamed("#id", "userid")

    # STEP 2: Data Cleaning - Filtering out nulls and invalid users (non-overlapping userIDs)
    filtered_music_user_timestamps_df = preprocess_music_data(music_df=music_user_timestamps_df, profiles_df=user_profiles_df)

    # STEP 3: Sessionize the data - define a global_session_key
    music_user_sessions_df = generate_session_info(df=filtered_music_user_timestamps_df, gap_threshold=1200)

    # STEP 4: Top 10 songs played in the top 50 longest sessions by track count
    top_sessions_keys = get_M_longest_sessions_by_track_count(df=music_user_sessions_df, top_m_sessions=50)
    top_songs_df = get_top_N_songs_by_count(top_sessions_keys, top_n_songs=10)

    # Save & Show the results
    top_songs_df.write.mode("overwrite").option("header", "true").csv(f"{DATASET_OUTPUT_DIRECTORY}top_songs_in_longest_sessions_by_track_count.csv")
    top_songs_df.show(truncate=False)

    # Stop the Spark session
    spark_session.stop()