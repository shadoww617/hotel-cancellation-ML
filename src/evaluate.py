import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
)

from logger import get_logger

logger = get_logger(__name__)


def evaluate(model, X_test, y_test):
    try:
        logger.info("Evaluation started")

        os.makedirs("artifacts/eval", exist_ok=True)

        # 1. CLASSIFICATION REPORT
        logger.info("Generating classification report")

        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred)

        logger.info("\n" + report)

        with open("artifacts/eval/classification_report.txt", "w") as f:
            f.write(report)

        # 2. CONFUSION MATRIX
        logger.info("Plotting confusion matrix")

        plt.figure(figsize=(6, 6))
        disp = ConfusionMatrixDisplay.from_estimator(
            model,
            X_test,
            y_test,
            display_labels=["Not Canceled", "Canceled"],
            cmap="Blues"
        )

        plt.title("Confusion Matrix – Hotel Booking Cancellation")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")

        cm_path = "artifacts/eval/confusion_matrix.png"
        plt.savefig(cm_path, dpi=200, bbox_inches="tight")
        plt.close()

        logger.info(f"Confusion matrix saved to {cm_path}")

        # 3. ROC CURVE
        logger.info("Plotting ROC curve")

        plt.figure(figsize=(6, 6))
        roc_disp = RocCurveDisplay.from_estimator(model, X_test, y_test)

        plt.plot([0, 1], [0, 1], "k--", label="Random Classifier")
        plt.legend()
        plt.title("ROC Curve – Cancellation Prediction")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")

        roc_path = "artifacts/eval/roc_curve.png"
        plt.savefig(roc_path, dpi=200, bbox_inches="tight")
        plt.close()

        logger.info(f"ROC curve saved to {roc_path}")

        # 4. FEATURE IMPORTANCE (XGBOOST)
        logger.info("Extracting feature importance from XGBoost")

        # Access trained XGBoost model inside pipeline
        xgb_model = model.named_steps["clf"]
        preprocessor = model.named_steps["pre"]

        # Get feature names after OneHotEncoding
        num_features = preprocessor.transformers_[0][2]
        cat_encoder = preprocessor.transformers_[1][1]
        cat_features = cat_encoder.get_feature_names_out(
            preprocessor.transformers_[1][2]
        )

        feature_names = list(num_features) + list(cat_features)

        importances = xgb_model.feature_importances_

        fi_df = pd.DataFrame({
            "feature": feature_names,
            "importance": importances
        }).sort_values(by="importance", ascending=False)

        fi_path = "artifacts/eval/feature_importance.csv"
        fi_df.to_csv(fi_path, index=False)

        logger.info(f"Feature importance saved to {fi_path}")

        # 5. FEATURE IMPORTANCE PLOT
        top_n = 15
        top_features = fi_df.head(top_n)

        plt.figure(figsize=(10, 6))
        plt.barh(
            top_features["feature"],
            top_features["importance"]
        )
        plt.gca().invert_yaxis()

        plt.title("Top 15 Most Important Features – XGBoost")
        plt.xlabel("Feature Importance Score")
        plt.ylabel("Feature Name")

        fi_plot_path = "artifacts/eval/top_feature_importance.png"
        plt.savefig(fi_plot_path, dpi=200, bbox_inches="tight")
        plt.close()

        logger.info(f"Top feature importance plot saved to {fi_plot_path}")

        logger.info("Top 10 Most Important Features:")
        logger.info("\n" + str(fi_df.head(10)))

        logger.info("Evaluation completed successfully")

    except Exception as e:
        logger.exception("Error occurred during evaluation")
        raise e
