import sys
import os
import io
import pytest
from datetime import datetime
from utils.constants import TESTS_DIRECTORY, TESTS_OUTPUT_DIRECTORY

if __name__ == "__main__":

    # Capture the results from UnitTests
    stdout_capture = io.StringIO()
    sys.stdout = stdout_capture

    # Any extra command line arguments passed to spark-submit will be forwarded to pytest
    exit_code = pytest.main([TESTS_DIRECTORY] + sys.argv[1:])

    # Extract string value from output. Reset stdout logging redirection
    sys.stdout = sys.__stdout__
    raw_logs = stdout_capture.getvalue()

    # Generate Report
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f"test_report_{current_time}.txt"

    text_report = f"""
        ================================================================================
        PYSPARK AUTOMATED TEST SUITE EXECUTION REPORT
        ================================================================================
        Runtime Node:    Docker Cluster Master Node (spark-master)
        Python Version:  {sys.version.split()[0]}
        ================================================================================

        STDOUT LOGS:
        --------------------------------------------------------------------------------
        {raw_logs}
        ================================================================================
        REPORT STATUS: {'GREEN (SUCCESS)' if exit_code == 0 else 'RED (FAILED)'}
        ================================================================================
    """

    os.makedirs(TESTS_OUTPUT_DIRECTORY, exist_ok=True)
    report_path = os.path.join(TESTS_OUTPUT_DIRECTORY, report_filename)
    with open(report_path, "w") as report_file:
        report_file.write(text_report)

    # Mirror the raw terminal execution metrics to screen so you see them live
    print(raw_logs)
    sys.exit(exit_code)