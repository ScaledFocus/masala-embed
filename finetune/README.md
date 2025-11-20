# Finetune

**CSV Format:** `text,image,dish_name`

Each row can have:

- Both `text` and `image` (multimodal)
- Only `text` (text-only)
- Only `image` (image-only)

Empty or NaN values are handled automatically.

## Download Images

```bash
INPUT_CSV=dataset.csv
OUTPUT_CSV=dataset.csv
IMAGE_DIR=images
python download_images.py
```

### Environment Variables

1. INPUT_CSV: Path to the input CSV file with image URLs (default: `dataset.csv`).
2. OUTPUT_CSV: Path to the output CSV file with local image paths (default: `dataset.csv`).
3. IMAGE_DIR: Directory to save downloaded images (default: `images`).

## Create Test Split

```bash
DATASET_CSV_NAME=dataset.csv
TRAIN_CSV_NAME=train.csv
TEST_CSV_NAME=test.csv
TEST_RATIO=0.2
python create_test_split.py
```

### Environment Variables

1. DATASET_CSV_NAME: Path to the dataset CSV file.
2. TRAIN_CSV_NAME: Path to the training CSV file.
3. TEST_CSV_NAME: Path to the test CSV file.
4. TEST_RATIO: Ratio of test data to total data.

## Finetune

```bash
MODEL_NAME=google/siglip2-base-patch16-224
TRAIN_CSV=train.csv
OUTPUT_DIR=finetuned_siglip
BATCH_SIZE=8
LEARNING_RATE=5e-6
NUM_EPOCHS=10
WARMUP_RATIO=0.1
DEVICE=cuda
python train.py
```

### Environment Variables

1. MODEL_NAME: Model to finetune (default: `google/siglip2-base-patch16-224`).
2. TRAIN_CSV: Path to training CSV file (default: `train.csv`).
3. OUTPUT_DIR: Directory to save finetuned model (default: `finetuned_siglip`).
4. BATCH_SIZE: Training batch size (default: `8`).
5. LEARNING_RATE: Learning rate for optimizer (default: `5e-6`).
6. NUM_EPOCHS: Number of training epochs (default: `10`).
7. WARMUP_RATIO: Ratio of warmup steps (default: `0.1`).
8. DEVICE: Device to use for training (default: `cuda` if available, else `cpu`).

## Evaluate

```bash
MODEL_PATH=finetuned_siglip
BENCHMARK_CSV=test.csv
OUTPUT_CSV=evaluation_results.csv
DEVICE=cuda
python evaluate.py
```

### Environment Variables

1. MODEL_PATH: Path to the model to evaluate (default: `google/siglip2-base-patch16-224`).
2. BENCHMARK_CSV: Path to the benchmark CSV file (default: `test.csv`).
3. OUTPUT_CSV: Path to the output CSV file (default: `evaluation_results.csv`).
4. DEVICE: Device to use for evaluation (default: `cuda` if available, else `cpu`).

## Dataset must have the following columns:

1. `text`: Text to generate image from.
2. `image`: Path to the image file.
3. `dish_name`: Label of the image.
