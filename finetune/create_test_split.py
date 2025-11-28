import os

import pandas as pd

DATASET_CSV_NAME = os.getenv("DATASET_CSV_NAME", "dataset.csv")
TRAIN_CSV_NAME = os.getenv("TRAIN_CSV_NAME", "train.csv")
TEST_CSV_NAME = os.getenv("TEST_CSV_NAME", "test.csv")
TEST_RATIO = float(os.getenv("TEST_RATIO", "0.2"))


def create_test_split(input_csv, train_csv, test_csv, test_ratio=0.2):
    df = pd.read_csv(input_csv)

    dishes = df["dish_name"].unique()

    train_rows = []
    test_rows = []

    for dish in dishes:
        dish_rows = df[df["dish_name"] == dish]
        n_rows = len(dish_rows)
        n_test = max(1, int(n_rows * test_ratio))

        shuffled = dish_rows.sample(frac=1, random_state=42)

        test_rows.append(shuffled.iloc[:n_test])
        train_rows.append(shuffled.iloc[n_test:])

    train_df = pd.concat(train_rows, ignore_index=True)
    test_df = pd.concat(test_rows, ignore_index=True)

    train_df.to_csv(train_csv, index=False)
    test_df.to_csv(test_csv, index=False)


if __name__ == "__main__":
    create_test_split(
        DATASET_CSV_NAME, TRAIN_CSV_NAME, TEST_CSV_NAME, test_ratio=TEST_RATIO
    )
