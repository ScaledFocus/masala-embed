# ESCI Dataset Annotation Tools

Flask-based web tools for efficiently annotating ESCI dataset entries with dual-mode support (CSV files and database).

## Features

### Two Annotation Modes
- **Regular Tool** (`app.py`): Single-record detailed annotation
- **Bulk Tool** (`app_bulk.py`): Multi-record compact annotation

### Dual Data Source Support
- **CSV Mode**: Direct file editing with immediate saving
- **Database Mode**: MLflow integration with human/AI label separation

### Core Features
- Clean display of consumable and query information
- One-click ESCI labeling (E/S/C/I buttons)
- Circular navigation with auto-advance
- Real-time progress tracking
- Comprehensive keyboard shortcuts
- Query text editing (database mode)
- Example and label deletion capabilities

## Usage

### Regular Annotation Tool
```bash
# Edit parameters in script, then run
./run_annotation_app.sh
```
Opens: http://localhost:5002

### Bulk Annotation Tool
```bash
# Edit parameters in script, then run
./run_bulk_annotation.sh
```
Opens: http://localhost:5003

### Configuration
Edit the run scripts to set:
- **Mode**: `"CSV"` or `"DATABASE"`
- **CSV file path** (for CSV mode)
- **MLflow run ID and labeler name** (for database mode)
- **Page size** (bulk tool only)

## Interface Differences

### Regular Tool (Single Record)
- **Side-by-side layout**: Consumable and query panels
- **Large displays**: Detailed view with metadata
- **Auto-advance**: Moves to next record after labeling
- **Skip navigation**: Jump to specific record numbers

### Bulk Tool (Multi-Record)
- **Grid layout**: Multiple records visible simultaneously
- **Compact cards**: Focused on consumable name and query
- **Click-to-focus**: Select any card for keyboard navigation
- **Page-based**: Configurable records per page

## ESCI Labels

- **🟢 E (Exact)**: Item precisely matches the query
- **🟡 S (Substitute)**: Item can replace the queried item
- **🟠 C (Complement)**: Item typically pairs with the queried item
- **🔴 I (Irrelevant)**: Item doesn't satisfy the query

## Keyboard Shortcuts

### Both Tools
- **Z** = Exact (E)
- **X** = Substitute (S)
- **C** = Complement (C)
- **V** = Irrelevant (I)
- **Esc** = Clear human label
- **Enter** = Copy AI label to human label (database mode)

### Regular Tool
- **← →** = Navigate between records (circular)

### Bulk Tool
- **Click card** = Focus for keyboard navigation
- **← → ↑ ↓** = Navigate focus between cards (circular)
- **Shift + ← →** = Navigate between pages

## Database Mode Features

- **Dual labels**: Separate display of AI and human labels
- **Query editing**: Click query text to edit with global deduplication
- **Example deletion**: Remove entire examples and orphaned queries
- **Label deletion**: Clear only your labels (preserves AI labels)
- **Copy AI labels**: Quickly adopt AI suggestions
- **MLflow integration**: Load examples from specific experiment runs

## Data Sources

### CSV Mode
Expects CSV with columns:
- `consumable_id`, `consumable_name`
- `query`, `query_filters` (JSON)
- `esci_label`, `generated_at`

### Database Mode
Loads from PostgreSQL with:
- **MLflow run filtering**: Specific experiment data
- **Labeler separation**: AI vs human annotations
- **Real-time updates**: Direct database persistence
- **Query deduplication**: Global query reuse