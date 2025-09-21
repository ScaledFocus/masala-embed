# Makefile for esci-dataset database operations
.PHONY: help clear-table load-consumables

# Default target
help:
	@echo "Available commands:"
	@echo "  make clear-table TABLE=<table_name>  - Clear all records from specified table"
	@echo "  make load-consumables                - Load consumables from CSV to database"
	@echo "  make help                           - Show this help message"
	@echo ""
	@echo "Examples:"
	@echo "  make clear-table TABLE=consumable"
	@echo "  make load-consumables INPUT=MM-Food-100K.csv THRESHOLD=0.9"

# Clear table target
clear-table:
ifndef TABLE
	@echo "Error: TABLE parameter is required"
	@echo "Usage: make clear-table TABLE=<table_name>"
	@exit 1
endif
	python database/scripts/clear_table.py --table $(TABLE) --confirm

# Load consumables target
load-consumables:
	python database/scripts/load_consumables.py \
		$(if $(INPUT),--input $(INPUT),) \
		$(if $(THRESHOLD),--threshold $(THRESHOLD),) \
		$(if $(BATCH_SIZE),--batch-size $(BATCH_SIZE),) \
		$(if $(DATA_FOLDER),--data-folder $(DATA_FOLDER),) \
		$(if $(DRY_RUN),--dry-run,)

# Load consumables with dry run
load-consumables-dry:
	python database/scripts/load_consumables.py --dry-run \
		$(if $(INPUT),--input $(INPUT),) \
		$(if $(THRESHOLD),--threshold $(THRESHOLD),) \
		$(if $(BATCH_SIZE),--batch-size $(BATCH_SIZE),) \
		$(if $(DATA_FOLDER),--data-folder $(DATA_FOLDER),)

# Example: make generate-queries ESCI=E LIMIT=100 BATCH_SIZE=10
# Example: make generate-queries ESCI=S LIMIT=50 BATCH_SIZE=5 OUTPUT_FORMAT=csv
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
		$(if $(MODEL),--model $(MODEL),)