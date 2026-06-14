FROM apache/spark:3.5.1

# Install Python dependencies (pandas and numpy) for PySpark jobs
USER root

RUN apt-get update && apt-get install -y \
    python3-pip && \
    rm -rf /var/lib/apt/lists/*

# Install heavy dependencies separately using pip's optimization flags
RUN pip3 install --no-cache-dir pandas scipy statsmodels
RUN pip3 install pytest

# Create a directory for Spark jobs & data. Set appropriate permissions for write access by the spark user.
RUN mkdir -p /opt/spark-jobs /opt/spark-data && \
    chown -R spark:spark /opt/spark-jobs /opt/spark-data

# Switch to the spark user and set the working directory
USER spark
WORKDIR /opt/spark-jobs