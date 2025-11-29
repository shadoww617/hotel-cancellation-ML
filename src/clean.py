import numpy as np
import pandas as pd
from logger import get_logger

logger = get_logger(__name__)


def clean_data(df):
    try:
        logger.info("Starting data cleaning and type conversion")

        if "Booking_ID" in df.columns:
            df.drop(columns=["Booking_ID"], inplace=True)
            logger.info("Dropped Booking_ID column")

        before = df.shape[0]
        df.drop_duplicates(inplace=True)
        after = df.shape[0]
        logger.info(f"Removed {before - after} duplicate rows")

        # Numerical columns
        num_cols = [
            "no_of_adults", "no_of_children", "no_of_weekend_nights",
            "no_of_week_nights", "lead_time", "no_of_previous_cancellations",
            "no_of_previous_bookings_not_canceled",
            "avg_price_per_room", "no_of_special_requests"
        ]

        for col in num_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        logger.info("Numeric type conversion completed")

        # Categorical columns
        cat_cols = [
            "type_of_meal_plan",
            "room_type_reserved",
            "market_segment_type",
            "booking_status",
            "repeated_guest"
        ]

        for col in cat_cols:
            if col in df.columns:
                df[col] = df[col].astype("category")

        logger.info("Categorical type conversion completed")

        # Date handling
        date_parts = {"arrival_year", "arrival_month", "arrival_date"}
        if date_parts.issubset(df.columns):
            df["arrival_date_full"] = pd.to_datetime(
                dict(
                    year=df["arrival_year"],
                    month=df["arrival_month"],
                    day=df["arrival_date"]
                ),
                errors="coerce"
            )
            logger.info("Arrival date converted to datetime")

        df["no_of_adults"] = df["no_of_adults"].clip(lower=0)
        df["no_of_children"] = df["no_of_children"].clip(lower=0)

        for col in df.select_dtypes(include=np.number).columns:
            df[col].fillna(df[col].median(), inplace=True)

        for col in df.select_dtypes(exclude=np.number).columns:
            df[col].fillna(df[col].mode()[0], inplace=True)

        logger.info("Missing value imputation completed")

        logger.info("Data cleaning and type conversion finished successfully")
        return df

    except Exception as e:
        logger.exception("Error during data cleaning and type conversion")
        raise e


def cap_outliers(df):
    try:
        logger.info("Starting outlier treatment")

        for col in ["avg_price_per_room", "lead_time"]:
            if col in df.columns:
                lower = df[col].quantile(0.01)
                upper = df[col].quantile(0.99)
                df[col] = df[col].clip(lower, upper)

        logger.info("Outlier treatment completed")
        return df

    except Exception as e:
        logger.exception("Error during outlier handling")
        raise e
