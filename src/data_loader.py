import os
import subprocess
import pandas as pd
from pathlib import Path

# Project root = parent of src
BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_RAW_DIR = BASE_DIR / "data" / "raw"


def download_predict_heart_disease_data(data_dir: Path | str | None = None):
    """
    Download Predicting Heart Disease dataset from Kaggle into data/raw.

    Notes
    -----
    - Uses absolute paths so notebooks do not create nested data folders.
    - Skips download if files already exist.
    - Surfaces Kaggle CLI errors with helpful guidance.
    """

    raw_dir = Path(data_dir) if data_dir else DEFAULT_RAW_DIR
    raw_dir.mkdir(parents=True, exist_ok=True)

    train_path = raw_dir / "train.csv"
    test_path = raw_dir / "test.csv"

    if train_path.exists() and test_path.exists():
        print("Dataset already exists locally.")
        return

    print(f"Downloading to {raw_dir} ...")
    cmd = [
        "kaggle", "competitions", "download",
        "-c", "playground-series-s6e2",
        "-p", str(raw_dir)
    ]

    try:
        # Avoid locale decode issues on Windows by keeping output as bytes
        subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=False,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "Kaggle CLI not found. Install with 'pip install kaggle' and restart the kernel/terminal."
        ) from exc
    except subprocess.CalledProcessError as exc:
        # Show stderr to help diagnose (missing API key, auth error, rate limit, etc.)
        raise RuntimeError(
            "Kaggle CLI failed. Ensure kaggle.json is in %USERPROFILE%/.kaggle and credentials are valid.\n"
            f"Command: {' '.join(cmd)}\n"
            f"Stdout (bytes): {exc.stdout}\n"
            f"Stderr (bytes): {exc.stderr}"
        ) from exc

    zip_path = raw_dir / "predicting-heart-disease.zip"
    if zip_path.exists():
        import zipfile
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(raw_dir)
        zip_path.unlink()

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError("Expected CSVs not found after unzip.")
    print("Dataset downloaded and extracted.")

def load_data(data_dir: Path | str | None = None):
    """Load training and test datasets from data/raw."""
    raw_dir = Path(data_dir) if data_dir else DEFAULT_RAW_DIR
    train_df = pd.read_csv(raw_dir / "train.csv")
    test_df = pd.read_csv(raw_dir / "test.csv")
    return train_df, test_df

if __name__ == "__main__":
    download_predict_heart_disease_data()
    train, test = load_data()
    print(f"Train shape: {train.shape}")
    print(f"Test shape: {test.shape}")
    print(f"Heart Disease rate: {train['Heart Disease'].mean():.2%}")