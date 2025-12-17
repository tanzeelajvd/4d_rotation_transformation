import os
import math
import numpy as np
import pandas as pd
from pandas.api.types import is_string_dtype
from sklearn.preprocessing import LabelEncoder

DATASETS_DIR = "datasets"
OUTPUT_DIR = "transformed_datasets"

os.makedirs(OUTPUT_DIR, exist_ok=True)

enc = LabelEncoder()
np.seterr(divide="ignore", invalid="ignore")


# ---------- Rotation matrix (same math as your code) ----------
def get_rotation_matrix(x_deg, y_deg):
    x = math.radians(x_deg)
    y = math.radians(y_deg)

    return np.array([
        [math.cos(y), -math.sin(y), 0, 0],
        [math.sin(y),  math.cos(y), 0, 0],
        [0, 0, math.cos(x), -math.sin(x)],
        [0, 0, math.sin(x),  math.cos(x)]
    ])


# ---------- Normalize column ----------
def normalize(col):
    return (col - col.min()) / (col.max() - col.min())


# ---------- Load + preprocess dataset ----------
def preprocess_dataset(path):
    df = pd.read_csv(path)

    for col in df.columns:
        if is_string_dtype(df[col]):
            df[col] = enc.fit_transform(df[col].astype(str))
        df[col] = normalize(df[col])

    return df


# ---------- Split into 4D column blocks ----------
def split_into_4d_blocks(df):
    blocks = []
    cols = df.columns.tolist()

    for i in range(0, len(cols), 4):
        block_cols = cols[i:i+4]
        block = df[block_cols].values
        blocks.append((block_cols, block))

    return blocks


# ---------- Find optimal rotation ----------
def find_best_rotation(arr):
    arr = arr.T
    max_var = -np.inf
    best_x, best_y = 0, 0

    for x in range(0, 360, 4):
        for y in range(0, 360, 4):
            R = get_rotation_matrix(x, y)
            rotated = R @ arr
            diff = arr - rotated
            var = np.max(np.var(diff.T, axis=0))

            if var > max_var:
                max_var = var
                best_x, best_y = x, y

    return best_x, best_y


# ---------- Apply rotation ----------
def rotate_block(arr, x, y):
    R = get_rotation_matrix(x, y)
    return (R @ arr.T).T


# ---------- Process one dataset ----------
def process_dataset(filename):
    print(f"\nProcessing dataset: {filename}")

    df = preprocess_dataset(os.path.join(DATASETS_DIR, filename))
    blocks = split_into_4d_blocks(df)

    transformed_blocks = []

    for cols, block in blocks:
        if block.shape[1] < 4:
            transformed_blocks.append(block)
            continue

        x, y = find_best_rotation(block)
        rotated = rotate_block(block, x, y)
        transformed_blocks.append(rotated)

        print(f"  Rotated columns {cols} with angles x={x}, y={y}")

    transformed = np.hstack(transformed_blocks)
    out_df = pd.DataFrame(transformed, columns=df.columns)

    out_path = os.path.join(OUTPUT_DIR, f"transformed_{filename}")

    from visualize import plot_pca_comparison

    plot_pca_comparison(
        df.values,
        transformed,
        title=f"{filename}: Original vs 4D Rotated"
    )

    out_df.to_csv(out_path, index=False)

    print(f"  Saved → {out_path}")


# ---------- Main ----------
def main():
    datasets = [
        f for f in os.listdir(DATASETS_DIR)
        if f.endswith(".csv")
    ]

    for ds in datasets:
        process_dataset(ds)


if __name__ == "__main__":
    main()
