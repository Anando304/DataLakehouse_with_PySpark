import pytest
from pyspark.sql import SparkSession, functions as F
from pyspark.sql import types as T
from utils.utils import generate_session_info, preprocess_music_data
from utils.metrics import (
    get_M_longest_sessions_by_track_count,
    get_M_longest_sessions_by_duration,
    get_top_N_songs_by_count,
    get_top_user_by_session_count,
    get_daily_session_counts
)
from datetime import datetime, timedelta

@pytest.fixture(scope="session")
def spark():
    """Fixture to create a local Spark session for testing."""
    return SparkSession.builder \
        .master("local[1]") \
        .appName("pytest-pyspark-utils-testing") \
        .getOrCreate()

def test_preprocess_music_data(spark):
    # Create mock music data
    music_schema = T.StructType([
        T.StructField("userid", T.StringType(), True),
        T.StructField("timestamp", T.StringType(), True),
        T.StructField("musicbrainz-artist-id", T.StringType(), True),
        T.StructField("artist-name", T.StringType(), True),
        T.StructField("musicbrainz-track-id", T.StringType(), True),
        T.StructField("track-name", T.StringType(), True)
    ])
    
    music_data = [
        ("user_1", "2023-01-01T00:00:00Z", "art_1", "Artist 1", "track_1", "Song 1"), # Valid
        (None, "2023-01-01T00:01:00Z", "art_1", "Artist 1", "track_2", "Song 2"),     # Null user
        ("user_1", "2023-01-01T00:02:00Z", "art_1", "Artist 1", None, "Song 3"),      # Null track ID
        ("user_2", "2023-01-01T00:03:00Z", "art_2", "Artist 2", "track_3", "Song 4"), # Doesn't exist in profile dataset below
    ]
    music_df = spark.createDataFrame(music_data, music_schema)

    # Create dummy profile data
    profile_schema = T.StructType([
        T.StructField("userid", T.StringType(), True)
    ])
    profile_data = [("user_1",)] # Only user_1 exists
    profiles_df = spark.createDataFrame(profile_data, profile_schema)

    # Execute
    result_df = preprocess_music_data(music_df, profiles_df)
    results = result_df.collect()

    # Assertions
    assert len(results) == 1
    assert results[0]["userid"] == "user_1"
    assert results[0]["musicbrainz-track-id"] == "track_1"

def test_generate_session_info(spark):
    schema = T.StructType([
        T.StructField("userid", T.StringType(), True),
        T.StructField("timestamp", T.TimestampType(), True),
        T.StructField("musicbrainz-track-id", T.StringType(), True),
        T.StructField("track-name", T.StringType(), True)
    ])

    base_time = datetime(2023, 1, 1, 12, 0, 0)
    
    # user_1: 3 tracks, two in same session, one in new session (gap > 20 mins)
    # user_2: 1 track
    data = [
        ("user_1", base_time, "t1", "Song 1"),
        ("user_1", base_time + timedelta(minutes=10), "t2", "Song 2"), # Same session
        ("user_1", base_time + timedelta(minutes=31), "t3", "Song 3"), # New session
        ("user_2", base_time, "t4", "Song 4")                          # Different user
    ]
    df = spark.createDataFrame(data, schema)

    # Execute (gap_threshold=1200 seconds = 20 mins)
    result_df = generate_session_info(df, gap_threshold=1200)
    results = result_df.orderBy("userid", "timestamp").collect()

    # Assertions
    assert len(results) == 4
    
    # Check user_1 sessions
    # result[0] and result[1] should have the same session key
    assert results[0]["global_session_key"] == results[1]["global_session_key"]
    # result[2] should have a different session key
    assert results[0]["global_session_key"] != results[2]["global_session_key"]
    
    # Verify global_session_key format
    assert results[0]["global_session_key"].startswith("user_1")
    assert results[3]["global_session_key"].startswith("user_2")

def test_generate_session_info_missing_columns(spark):
    # Test error handling for missing columns
    invalid_df = spark.createDataFrame([("val",)], ["some_col"])
    with pytest.raises(ValueError, match="Input DataFrame must contain 'timestamp' and 'userid' columns"):
        generate_session_info(invalid_df)

def test_get_M_longest_sessions_by_track_count(spark):
    schema = T.StructType([
        T.StructField("global_session_key", T.StringType(), True),
        T.StructField("track-name", T.StringType(), True)
    ])
    # Session s1: 3 tracks, s2: 2 tracks, s3: 1 track
    data = [
        ("s1", "song1"), ("s1", "song2"), ("s1", "song3"),
        ("s2", "song1"), ("s2", "song2"),
        ("s3", "song1")
    ]
    df = spark.createDataFrame(data, schema)
    
    # Get top 2 sessions (s1 and s2)
    result_df = get_M_longest_sessions_by_track_count(df, top_m_sessions=2)
    results = result_df.collect()
    
    session_keys = {row["global_session_key"] for row in results}
    assert "s1" in session_keys
    assert "s2" in session_keys
    assert "s3" not in session_keys

def test_get_M_longest_sessions_by_duration(spark):
    schema = T.StructType([
        T.StructField("global_session_key", T.StringType(), True),
        T.StructField("timestamp", T.LongType(), True)
    ])
    # s1: 100s, s2: 300s, s3: 50s
    data = [
        ("s1", 1000), ("s1", 1100),
        ("s2", 2000), ("s2", 2300),
        ("s3", 3000), ("s3", 3050)
    ]
    df = spark.createDataFrame(data, schema)
    
    # Get top 2 sessions (s2 and s1)
    result_df = get_M_longest_sessions_by_duration(df, top_m_sessions=2)
    results = result_df.collect()
    
    session_keys = {row["global_session_key"] for row in results}
    assert "s2" in session_keys
    assert "s1" in session_keys
    assert "s3" not in session_keys

def test_get_top_N_songs_by_count(spark):
    schema = T.StructType([
        T.StructField("musicbrainz-track-id", T.StringType(), True),
        T.StructField("track-name", T.StringType(), True)
    ])
    data = [("t1", "S1"), ("t1", "S1"), ("t1", "S1"), ("t2", "S2")]
    df = spark.createDataFrame(data, schema)
    
    result_df = get_top_N_songs_by_count(df, top_n_songs=1)
    results = result_df.collect()
    
    assert len(results) == 1
    assert results[0]["musicbrainz-track-id"] == "t1"
    assert results[0]["play_count"] == 3

def test_get_top_user_by_session_count(spark):
    schema = T.StructType([
        T.StructField("userid", T.StringType(), True),
        T.StructField("global_session_key", T.StringType(), True)
    ])
    # u1: 2 unique sessions, u2: 1 unique session
    data = [
        ("u1", "s1"), ("u1", "s1"), # Duplicate tracks in same session
        ("u1", "s2"),
        ("u2", "s3")
    ]
    df = spark.createDataFrame(data, schema)
    
    result_df = get_top_user_by_session_count(df)
    results = result_df.collect()
    
    assert len(results) == 1
    assert results[0]["userid"] == "u1"
    assert results[0]["session_count"] == 2

def test_get_daily_session_counts(spark):
    schema = T.StructType([
        T.StructField("userid", T.StringType(), True),
        T.StructField("timestamp", T.TimestampType(), True),
        T.StructField("global_session_key", T.StringType(), True)
    ])
    
    day1 = datetime(2023, 1, 1, 10, 0, 0)
    day2 = datetime(2023, 1, 2, 10, 0, 0)
    
    data = [
        ("u1", day1, "s1"), ("u1", day1, "s2"), # 2 sessions on day 1
        ("u1", day2, "s3"),                     # 1 session for u1 on day 2
        ("u2", day2, "s4")                      # 1 session for u2 on day 2
    ]
    df = spark.createDataFrame(data, schema)
    
    # Test case 1: Total counts across all users
    res_all = get_daily_session_counts(df).collect()
    assert len(res_all) == 2
    assert res_all[0]["total_session_count"] == 2 # s1, s2
    assert res_all[1]["total_session_count"] == 2 # s3, s4
    
    # Test case 2: Filtered for a specific user
    res_u1 = get_daily_session_counts(df, userid="u1").collect()
    assert len(res_u1) == 2
    assert res_u1[0]["total_session_count"] == 2 # s1, s2
    assert res_u1[1]["total_session_count"] == 1 # s3 only