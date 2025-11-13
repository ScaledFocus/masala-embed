import pandas as pd


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

    print(f"Total rows: {len(df)}")
    print(f"Train rows: {len(train_df)} ({len(train_df) / len(df) * 100:.1f}%)")
    print(f"Test rows: {len(test_df)} ({len(test_df) / len(df) * 100:.1f}%)")
    print(f"Total dishes: {len(dishes)}")
    print(f"Train saved to: {train_csv}")
    print(f"Test saved to: {test_csv}")


if __name__ == "__main__":
    create_test_split("finetuning_dataset.csv", "train.csv", "test.csv", test_ratio=0.2)
