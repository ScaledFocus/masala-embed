# Makefile for esci-dataset database operations
.PHONY: help clear-table load-consumables generate-queries approve-runs migrate-data

# Default target
help:
	@echo "Available commands:"
	@echo "  make clear-table TABLE=<table_name>  - Clear all records from specified table"
	@echo "  make load-consumables                - Load consumables from CSV to database"
	@echo "  make generate-queries                - Generate food delivery queries using AI"
	@echo "  make approve-runs RUNS=<run_ids>     - Approve MLflow runs for migration"
	@echo "  make migrate-data [EXPERIMENT=<name>]- Migrate approved synthetic data to database"
	@echo "  make help                           - Show this help message"
	@echo ""
	@echo "Examples:"
	@echo "  make clear-table TABLE=consumable"
	@echo "  make clear-table TABLE=query_dataset"
	@echo "  make load-consumables INPUT=MM-Food-100K.csv THRESHOLD=0.9"
	@echo "  make load-consumables INPUT=data/food.csv THRESHOLD=0.8 BATCH_SIZE=500"
	@echo "  make load-consumables-dry INPUT=MM-Food-100K.csv"
	@echo "  make load-consumables-dry INPUT=data/food.csv THRESHOLD=0.9 BATCH_SIZE=200"
	@echo "  make generate-queries ESCI=E LIMIT=100 BATCH_SIZE=10"
	@echo "  make generate-queries ESCI=S LIMIT=200 BATCH_SIZE=20 DIETARY_FLAG=1 OUTPUT_FORMAT=csv OUTPUT_PATH=output/custom.csv TEMPERATURE=1.0 QUERIES_PER_ITEM=3 MODEL=gpt-5 QUERY_EXAMPLES=prompts/query_generation/examples.txt"
	@echo "  make approve-runs RUNS='abc123 def456 ghi789'"
	@echo "  make migrate-data EXPERIMENT=initial-generation-test"
	@echo "  make migrate-data"

# Example: make clear-table TABLE=consumable
# Example: make clear-table TABLE=query_dataset
# Clear table target
clear-table:
ifndef TABLE
	@echo "Error: TABLE parameter is required"
	@echo "Usage: make clear-table TABLE=<table_name>"
	@exit 1
endif
	python database/scripts/clear_table.py --table $(TABLE) --confirm

# Example: make load-consumables INPUT=MM-Food-100K.csv THRESHOLD=0.9
# Example: make load-consumables INPUT=data/food.csv THRESHOLD=0.8 BATCH_SIZE=500
# Example: make load-consumables INPUT=data/food.csv DATA_FOLDER=./data BATCH_SIZE=1000
# Load consumables target
load-consumables:
	python database/scripts/load_consumables.py \
		$(if $(INPUT),--input $(INPUT),) \
		$(if $(THRESHOLD),--threshold $(THRESHOLD),) \
		$(if $(BATCH_SIZE),--batch-size $(BATCH_SIZE),) \
		$(if $(DATA_FOLDER),--data-folder $(DATA_FOLDER),) \
		$(if $(DRY_RUN),--dry-run,)

# Example: make load-consumables-dry INPUT=MM-Food-100K.csv
# Example: make load-consumables-dry INPUT=data/food.csv THRESHOLD=0.9 BATCH_SIZE=200
# Load consumables with dry run
load-consumables-dry:
	python database/scripts/load_consumables.py --dry-run \
		$(if $(INPUT),--input $(INPUT),) \
		$(if $(THRESHOLD),--threshold $(THRESHOLD),) \
		$(if $(BATCH_SIZE),--batch-size $(BATCH_SIZE),) \
		$(if $(DATA_FOLDER),--data-folder $(DATA_FOLDER),)

# Example: make generate-queries ESCI=E LIMIT=100 BATCH_SIZE=10
# Example: make generate-queries ESCI=S LIMIT=200 BATCH_SIZE=20 DIETARY_FLAG=1 OUTPUT_FORMAT=csv OUTPUT_PATH=output/custom.csv TEMPERATURE=1.0 QUERIES_PER_ITEM=3 MODEL=gpt-5 QUERY_EXAMPLES=prompts/query_generation/examples.txt
# Generate food delivery queries using AI
generate-queries:
ifndef ESCI
	@echo "Error: ESCI parameter is required (E, S, C, or I)"
	@echo "Usage: make generate-queries ESCI=<E|S|C|I> [LIMIT=<num>] [BATCH_SIZE=<num>] [OPTIONS...]"
	@exit 1
endif
	uv run python src/data_generation/initial_generation.py \
		--esci_label $(ESCI) \
		$(if $(LIMIT),--limit $(LIMIT),) \
		$(if $(BATCH_SIZE),--batch_size $(BATCH_SIZE),) \
		$(if $(DIETARY_FLAG),--dietary_flag,) \
		$(if $(OUTPUT_FORMAT),--output_format $(OUTPUT_FORMAT),) \
		$(if $(OUTPUT_PATH),--output_path $(OUTPUT_PATH),) \
		$(if $(TEMPERATURE),--temperature $(TEMPERATURE),) \
		$(if $(QUERIES_PER_ITEM),--queries_per_item $(QUERIES_PER_ITEM),) \
		$(if $(MODEL),--model $(MODEL),) \
		$(if $(QUERY_EXAMPLES),--query_examples $(QUERY_EXAMPLES),)

# Example: make approve-runs RUNS='abc123 def456 ghi789'
# Approve MLflow runs for migration
approve-runs:
ifndef RUNS
	@echo "Error: RUNS parameter is required"
	@echo "Usage: make approve-runs RUNS='<run_id1> [run_id2] [run_id3]'"
	@echo "Example: make approve-runs RUNS='abc123 def456 ghi789'"
	@exit 1
endif
	uv run python src/mlflow_wrapper/approve_run.py $(RUNS)

# Example: make migrate-data EXPERIMENT=initial-generation-test
# Example: make migrate-data
# Migrate approved synthetic data to database
migrate-data:
	@echo "Migrating approved synthetic data..."
	uv run python database/scripts/migrate_synthetic_data.py \
		$(if $(EXPERIMENT),--experiment-name $(EXPERIMENT),--all-approved) \
		--confirm

hash:
	git rev-parse --short HEAD