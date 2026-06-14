# Variables
MASTER_CONTAINER = spark-master
SPARK_SUBMIT = /opt/spark/bin/spark-submit
MASTER_URL = spark://spark-master:7077
JOBS_DIR = /opt/spark-jobs

.PHONY: up down restart status logs master_shell run nuke

# Start the Spark cluster [Orbstack, DockerDesktop, etc. must be running first]
up:
	docker compose up -d

# Stop and tear down the cluster containers
down:
	docker compose down

# Hard restart the cluster and force-rebuild the Dockerfile if changes were made
restart:
	docker compose down
	docker compose up -d --build

# View the status of the cluster containers
status:
	docker compose ps

# Stream live logs from the worker container to see print statements
logs:
	docker compose logs -f spark-worker

# Pop directly inside the terminal of the spark-master container
master_shell:
	docker compose exec -it $(MASTER_CONTAINER) bash

run_job2:
	docker compose exec $(MASTER_CONTAINER) $(SPARK_SUBMIT) --master $(MASTER_URL) $(JOBS_DIR)/problem_2.py

run_job3:
	docker compose exec $(MASTER_CONTAINER) $(SPARK_SUBMIT) --master $(MASTER_URL) $(JOBS_DIR)/problem_3.py

# Execute a specific PySpark job. Usage: make run_job job=problem_2/problem_2.py
run_job:
	@if [ -z "$(job)" ]; then \
		echo "Error: Please specify a job file. Example: make run_job job=problem_2.py"; \
		exit 1; \
	fi
	docker compose exec $(MASTER_CONTAINER) $(SPARK_SUBMIT) --master $(MASTER_URL) $(JOBS_DIR)/$(job)

# Run all Unit Tests
run_tests:
	docker compose exec $(MASTER_CONTAINER) env PYTHONPATH="$(JOBS_DIR)" $(SPARK_SUBMIT) --master local[*] $(JOBS_DIR)/tests/run_tests.py

# Wipe out unused Docker caches, networks, and volumes
nuke:
	docker compose down
	docker system prune -a --volumes -f