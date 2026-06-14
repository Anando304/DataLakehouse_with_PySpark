"""
This module contains the metric functions for Problem 2

Eg;;
    - What are the top 10 songs played in the top 50 longest sessions by tracks count?
        - top_sessions_keys = get_M_longest_sessions_by_track_count(df, top_m_sessions=50)
        - top_songs_df = get_top_N_songs_by_count(top_sessions_keys, top_n_songs=10)
"""

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

def get_M_longest_sessions_by_track_count(df: DataFrame, top_m_sessions: int) -> DataFrame:
    """
    Filters the DataFrame to include only records from the top M longest sessions by track count.

    Args:
        df (DataFrame): Input DataFrame with session info.
        top_m_sessions (int): Number of top sessions to filter.

    Returns:
        DataFrame: Records belonging to the top M sessions.
    """
    session_stats_df = df.groupBy("global_session_key") \
                        .agg(F.count("*").alias("track_count"))

    top_sessions_keys = session_stats_df.orderBy(F.desc("track_count")) \
                                       .limit(top_m_sessions) \
                                       .select("global_session_key")

    return df.join(F.broadcast(top_sessions_keys), "global_session_key", "inner")


def get_M_longest_sessions_by_duration(df: DataFrame, top_m_sessions: int) -> DataFrame:
    """
    Filters the DataFrame to include only records from the top M longest sessions by duration.

    Args:
        df (DataFrame): Input DataFrame with session info and timestamps.
        top_m_sessions (int): Number of top sessions to filter.

    Returns:
        DataFrame: Records belonging to the top M sessions.
    """
    session_stats_df = df.groupBy("global_session_key").agg(
        (F.max("timestamp").cast("long") - F.min("timestamp").cast("long")).alias("session_duration_sec")
    )

    top_sessions_keys = session_stats_df.orderBy(F.desc("session_duration_sec")) \
                                       .limit(top_m_sessions) \
                                       .select("global_session_key")

    return df.join(F.broadcast(top_sessions_keys), "global_session_key", "inner")


def get_top_N_songs_by_count(df: DataFrame, top_n_songs: int) -> DataFrame:
    """
    Aggregates the track play counts and returns the top N songs.

    Args:
        df (DataFrame): Input DataFrame with track info.
        top_n_songs (int): Number of top songs to return.

    Returns:
        DataFrame: Top N songs with play counts.
    """
    top_songs_df = df.groupBy("musicbrainz-track-id", "track-name") \
                    .agg(F.count("*").alias("play_count")) \
                    .orderBy(F.desc("play_count")) \
                    .limit(top_n_songs)

    return top_songs_df.select("musicbrainz-track-id", "track-name", "play_count")

def get_top_user_by_session_count(df: DataFrame) -> DataFrame:
    """
    Identifies the user who has the highest number of unique sessions.

    Args:
        df (DataFrame): Input DataFrame with session info (userid, global_session_key).

    Returns:
        DataFrame: A single-row DataFrame containing the 'userid' and 'session_count'.
    """
    return df.groupBy("userid") \
             .agg(F.countDistinct("global_session_key").alias("session_count")) \
             .orderBy(F.desc("session_count")) \
             .limit(1)

def get_daily_session_counts(df: DataFrame, userid: str = None) -> DataFrame:
    """
    Aggregates the number of unique sessions per day.
    Optionally, if userid is provided, it will filter for that specific user.

    Args:
        df (DataFrame): Input DataFrame with session info and timestamps.
        userid (str) [OPTIONAL]: The specific user ID to filter for.

    Returns:
        DataFrame: Time series with 'date' and 'total_session_count'.
    """
    if userid:
        df = df.filter(F.col("userid") == userid)

    if "date" not in df.columns:
        df = df.withColumn("date", F.to_date("timestamp"))

    return df.groupBy("date") \
             .agg(F.countDistinct("global_session_key").alias("total_session_count")) \
             .orderBy("date")