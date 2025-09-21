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