import pandas as pd
import os
from logger import get_logger

logger = get_logger(__name__)

def load_data(path):
    try:
        logger.info(f"Loading dataset from {path}")

        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset not found at: {path}")

        df = pd.read_csv(path)

        logger.info(f"Dataset loaded successfully with shape {df.shape}")
        return df

    except Exception as e:
        logger.exception("Error while loading dataset")
        raise e
