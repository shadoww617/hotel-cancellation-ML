import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from logger import get_logger

logger = get_logger(__name__)


def run_eda(df):
    try:
        logger.info("EDA started")

        os.makedirs("artifacts/eda", exist_ok=True)

        # 1. DATASET SUMMARY
        logger.info("Generating dataset summary")

        logger.info(f"Dataset Shape: {df.shape}")

        logger.info(f"Data Types:\n{df.dtypes}")

        missing = df.isna().sum()
        logger.info(f"Missing Values:\n{missing}")

        duplicates = df.duplicated().sum()
        logger.info(f"Duplicate Rows: {duplicates}")

        info = df.info()
        logger.info(f"Dataset Info:\n{info}")

        df.describe(include="all").to_csv("artifacts/eda/dataset_summary.csv")
        logger.info("Dataset summary saved to artifacts/eda/dataset_summary.csv")

        # 2. NUMERIC DISTRIBUTIONS (LABELED)
        logger.info("Generating numeric feature distributions")

        numeric_cols = df.select_dtypes(include=np.number).columns

        df[numeric_cols].hist(figsize=(18, 12), bins=30)
        plt.suptitle("Distribution of Numerical Features", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        dist_path = "artifacts/eda/numeric_distributions.png"
        plt.savefig(dist_path, dpi=200, bbox_inches="tight")
        plt.close()

        logger.info(f"Numeric distributions saved to {dist_path}")

        # 3. TARGET CLASS DISTRIBUTION
        if "booking_status" in df.columns:
            logger.info("Generating booking status distribution")

            plt.figure(figsize=(6, 4))
            sns.countplot(x="booking_status", data=df)

            plt.title("Distribution of Booking Status")
            plt.xlabel("Booking Status")
            plt.ylabel("Number of Bookings")

            target_path = "artifacts/eda/target_distribution.png"
            plt.savefig(target_path, dpi=200, bbox_inches="tight")
            plt.close()

            logger.info(f"Target distribution saved to {target_path}")

        # 4. CORRELATION HEATMAP
        logger.info("Generating correlation heatmap")

        corr = df[numeric_cols].corr()

        plt.figure(figsize=(14, 10))
        sns.heatmap(
            corr,
            cmap="coolwarm",
            center=0,
            square=True,
            linewidths=0.5
        )

        plt.title("Correlation Heatmap of Numerical Features")
        plt.xlabel("Features")
        plt.ylabel("Features")

        corr_path = "artifacts/eda/correlation_heatmap.png"
        plt.savefig(corr_path, dpi=200, bbox_inches="tight")
        plt.close()

        logger.info(f"Correlation heatmap saved to {corr_path}")

        # 5. LEAD TIME vs BOOKING STATUS
        if {"lead_time", "booking_status"}.issubset(df.columns):
            logger.info("Generating Lead Time vs Booking Status plot")

            plt.figure(figsize=(8, 5))
            sns.boxplot(x="booking_status", y="lead_time", data=df)

            plt.title("Lead Time vs Booking Status")
            plt.xlabel("Booking Status")
            plt.ylabel("Lead Time (Days)")

            leadtime_path = "artifacts/eda/leadtime_vs_status.png"
            plt.savefig(leadtime_path, dpi=200, bbox_inches="tight")
            plt.close()

            logger.info(f"Lead time pattern saved to {leadtime_path}")

        # 6. PRICE vs BOOKING STATUS
        if {"avg_price_per_room", "booking_status"}.issubset(df.columns):
            logger.info("Generating Price vs Booking Status plot")

            plt.figure(figsize=(8, 5))
            sns.boxplot(x="booking_status", y="avg_price_per_room", data=df)

            plt.title("Average Room Price vs Booking Status")
            plt.xlabel("Booking Status")
            plt.ylabel("Average Price per Room")

            price_path = "artifacts/eda/price_vs_status.png"
            plt.savefig(price_path, dpi=200, bbox_inches="tight")
            plt.close()

            logger.info(f"Price pattern saved to {price_path}")

        # 7. AUTOMATED DATA ISSUE DETECTION
        logger.info("Running automated data issue detection")

        issues = []

        if duplicates > 0:
            issues.append(f"{duplicates} duplicate rows found")

        high_missing = missing[missing > 0.3 * len(df)]
        if not high_missing.empty:
            issues.append(f"Columns with >30% missing values:\n{high_missing}")

        for col in numeric_cols:
            if (df[col] < 0).any():
                issues.append(f"Negative values detected in column: {col}")

            if df[col].skew() > 2:
                issues.append(f"Highly skewed column (possible outliers): {col}")

        issue_path = "artifacts/eda/data_issues.txt"
        with open(issue_path, "w") as f:
            if not issues:
                f.write("No major data issues detected.\n")
            else:
                for i in issues:
                    f.write(i + "\n")

        logger.info(f"Data issues logged to {issue_path}")

        logger.info("EDA completed successfully")

    except Exception as e:
        logger.exception("Error occurred during EDA")
        raise e
