from data_loader import load_data
from clean import clean_data, cap_outliers
from feature_engineering import add_features
from train import train_models
from evaluate import evaluate
from eda import run_eda
from logger import get_logger

import warnings
warnings.filterwarnings("ignore")
 
logger = get_logger(__name__)

def main():
    try:
        logger.info("========== PIPELINE STARTED ==========")

        logger.info("Loading dataset")
        df = load_data("data/Hotel Reservations.csv")

        logger.info("Running EDA")
        run_eda(df)

        logger.info("Cleaning data")
        df = clean_data(df)

        logger.info("Handling outliers")
        df = cap_outliers(df)

        logger.info("Feature engineering")
        df = add_features(df)

        logger.info("Preparing target variable")

        df.rename(columns={"booking_status": "target"}, inplace=True)
        df["target"] = df["target"].map({"Canceled": 1, "Not_Canceled": 0})

        X = df.drop(columns=["target"])
        y = df["target"]

        logger.info("Training models (LogReg, RandomForest, XGBoost)")
        best_model, X_test, y_test = train_models(X, y)

        logger.info("Evaluating best model")
        evaluate(best_model, X_test, y_test)

        logger.info("========== PIPELINE COMPLETED SUCCESSFULLY ==========")

    except Exception as e:
        logger.critical("========== PIPELINE FAILED ==========")
        logger.exception("Fatal error in pipeline")
        raise e


if __name__ == "__main__":
    main()
