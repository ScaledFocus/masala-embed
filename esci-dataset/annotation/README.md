# ESCI Dataset Annotation Tool

A Flask-based web tool for efficiently relabeling ESCI dataset entries.

## Features

- Clean side-by-side display of consumable and query
- One-click ESCI relabeling (E/S/C/I buttons)
- Auto-advance to next item after labeling
- Skip to any record by number
- Real-time progress tracking
- Immediate CSV saving (no downloads needed)
- Keyboard shortcuts for fast annotation

## Usage

```bash
# Using the convenience script (recommended)
./run_annotation_app.sh output/queries_I_batch100_limit200_v2_20250925_112245.csv

# Or run directly
python annotation/app.py output/queries_I_batch100_limit200_v2_20250925_112245.csv
```

Then open http://localhost:5000 in your browser.

## Interface

- **Left Panel**: Shows consumable ID and name
- **Right Panel**: Shows the search query and generation timestamp
- **Current Label**: Prominently displays the current ESCI label
- **Skip to Record**: Input field to jump to any record number
- **Progress Bar**: Visual progress with current/total count

## Controls

- **🟢 E (Exact)**: Item precisely matches the query
- **🟡 S (Substitute)**: Item can replace the queried item
- **🟠 C (Complement)**: Item typically pairs with the queried item
- **🔴 I (Irrelevant)**: Item doesn't satisfy the query

- **◀️ Previous / ▶️ Next**: Manual navigation
- **Skip to Record**: Type number + Enter/Go to jump to specific record

## Keyboard Shortcuts

- **Z** = Exact (E)
- **X** = Substitute (S)
- **C** = Complement (C)
- **V** = Irrelevant (I)
- **← →** = Navigate between records

## Auto-Features

- **Auto-advance**: Each label button moves to next item
- **Auto-save**: Changes saved immediately to CSV
- **Real-time updates**: No manual saving required

## File Format

Expects CSV with columns:
- `consumable_id`
- `consumable_name`
- `query`
- `query_filters` (JSON)
- `esci_label`
- `generated_at`