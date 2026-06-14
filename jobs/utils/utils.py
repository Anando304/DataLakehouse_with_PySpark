"""
This module contains utility functions for the PySpark Music Processor Pipelines:

Functionality include:
    - loading datasets
    - generating session information for user song play data. 
"""

from pyspark.sql.types import StructType, StructField, StringType
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql import DataFrame


def load_dataset(spark_session: SparkSession, data_path: str, schema: StructType = None) -> DataFrame:
    """
    Load the dataset into a PySpark DataFrame.

    Args:
        spark_session (SparkSession): The Spark session to use for loading data.
        data_path (str): The base path where the datasets are located.
        schema (StructType): Optional schema for the DataFrame. If None, the schema will be inferred.

    Returns:
        PySpark DataFrame containing the loaded dataset.
    """
    if schema:
        return spark_session.read.csv(data_path, header=False, sep="\u0009", schema=schema)
    
    return spark_session.read.csv(data_path, header=True, sep="\u0009")
    
    

def generate_session_info(df: DataFrame, gap_threshold: int = 1200)->DataFrame:
    """
    Generate session information for each user based on the provided gap_threshold(Default=20 mins) between song play timestamps.

    Args:
        df (DataFrame): The input DataFrame containing user song play data.
        gap_threshold (int): The time gap in seconds to determine the start of a new session. Default is 1200 seconds (20 minutes).

    Returns:
        PySpark DataFrame containing the original data WITH global_session_key column.
    """

    if "timestamp" not in df.columns or "userid" not in df.columns:
        raise ValueError("Input DataFrame must contain 'timestamp' and 'userid' columns for sessionization.")


    # Determine the time gap(second) between current timestamp and previous timestamp. Using this, flag new session: if greater than gap_threshold OR if first track for the user.
    window_spec = Window.partitionBy(F.col("userid")).orderBy(F.unix_timestamp(F.col("timestamp")))
    current_epoch = F.unix_timestamp(F.col("timestamp"))
    prev_epoch = F.unix_timestamp(F.lag("timestamp").over(window_spec))

    df_with_timestamp_gap = df.withColumn(
        "time_gap_sec", 
        current_epoch - prev_epoch
    )
    
    df_with_session_flags = df_with_timestamp_gap.withColumn(
        "new_session_flag", 
        F.when(F.col("time_gap_sec") > gap_threshold, 1)
        .otherwise(F.when(prev_epoch.isNull(), 1).otherwise(0))
    )
    
    session_window = Window.partitionBy("userid").orderBy("timestamp").rowsBetween(Window.unboundedPreceding, Window.currentRow)
    df_with_session_ids = df_with_session_flags.withColumn("session_id", F.sum("new_session_flag").over(session_window))

    # Generate Unique global session ID
    df_with_global_session_key = df_with_session_ids.withColumn("global_session_key", F.concat_ws("_", F.col("userid"), F.col("session_id")))

    return df_with_global_session_key.select("userid", "timestamp", "musicbrainz-track-id", "track-name", "global_session_key")



def preprocess_music_data(music_df: DataFrame, profiles_df: DataFrame) -> DataFrame:
    """
    Cleanses raw input logs of music_df and filters listening history to include only valid, 
    matching user profiles.

    Args:
        music_df (DataFrame): Raw listening history log DataFrame.
        profiles_df (DataFrame): Raw user profiles DataFrame.

    Returns:
        DataFrame: Cleansed and structurally validated listening history.
    """
    # 1. Row-level data quality constraints
    music_clean = music_df.filter(
        F.col("userid").isNotNull() & 
        F.col("musicbrainz-track-id").isNotNull() & 
        F.col("track-name").isNotNull()
    )
    
    profiles_clean = profiles_df.filter(F.col("userid").isNotNull())

    # 2. Retain only overlapping music userIDs - use broadcast join since profiles is small 
    music_clean_valid_users = music_clean.join(
        F.broadcast(profiles_clean.select("userid")), 
        on="userid", 
        how="inner"
    )

    return music_clean_valid_users.select(
        "userid", "timestamp", "musicbrainz-artist-id", "artist-name", "musicbrainz-track-id", "track-name"
    )