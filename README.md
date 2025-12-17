# 4D Rotation Transformation

This project implements a **4D rotation–based data transformation** method in which numerical features of a dataset are grouped into four-dimensional blocks and transformed using an orthogonal 4D rotation matrix. The transformation preserves geometric properties such as variance and distance while changing the orientation of the data in feature space for analysis and experimentation.

The transformation is applied **independently to each dataset**, ensuring that datasets are not mixed and that their internal structure remains intact.

---

## Method Overview

For a given dataset, the transformation follows these steps:

1. Numerical features are grouped into **blocks of four dimensions**.
2. Categorical features (if present) are encoded, and all features are normalized.
3. A **4D orthogonal rotation matrix** is applied to each block.
4. Rotation angles are selected using a **variance-based search strategy**.
5. All rotated blocks are recombined to form the transformed dataset.

This approach preserves variance magnitude and pairwise distances while re-orienting the feature space.

---

## Datasets Used

The method is demonstrated on the following datasets:

- `seeds_dataset.csv`
- `bank-full.csv`
- `forestfires.csv`
- `HCV-Egy-Data.csv`
- `Data_User_Modeling_Dataset.csv`

Each dataset is processed independently.

---

## Repository Structure

```

4d_rotation_transformation/
├── datasets/                 # Input datasets
├── transformed_datasets/     # Output transformed datasets
├── rotation_4d.py            # Main implementation of the 4D rotation transformation
├── visualize.py              # Visualization utilities
├── environment.yml           # Conda environment specification
├── requirements.txt          # Pip dependency list
└── README.md

```

---

## Output

For each input dataset, a corresponding transformed dataset is generated and saved in the `transformed_datasets/` directory.

Example output files:

```

transformed_datasets/
├── transformed_seeds_dataset.csv
├── transformed_bank-full.csv
├── transformed_forestfires.csv
├── transformed_HCV-Egy-Data.csv
└── transformed_Data_User_Modeling_Dataset.csv

````

---

## Environment Setup

Using Conda (recommended):

```bash
conda env create -f environment.yml
conda activate rotation-4d
````

Using pip:

```bash
pip install -r requirements.txt
```

---

## Running the Code

From the repository root, run:

```bash
python rotation_4d.py
```

All transformed datasets will be saved automatically to the `transformed_datasets/` directory.

---

## Notes

* The transformation is deterministic for a given dataset and configuration.
* This method performs a geometric transformation and does not reduce dimensionality.
* Visualization utilities are provided for analysis and comparison only.

---

## License

This project is intended for academic and research use.

```
```
