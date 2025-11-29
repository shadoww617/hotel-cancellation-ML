import pandas as pd
from logger import get_logger

logger = get_logger(__name__)

def add_features(df):
    try:
        logger.info("Starting feature engineering")

        df["total_stay_nights"] = df["no_of_weekend_nights"] + df["no_of_week_nights"]

        df["total_guests"] = df["no_of_adults"] + df["no_of_children"]
        df["total_guests"].replace(0, 1, inplace=True)

        bins = [-1, 30, 90, 9999]
        labels = ["short", "medium", "long"]
        df["lead_time_category"] = pd.cut(df["lead_time"], bins=bins, labels=labels)

        df["avg_adr_per_person"] = df["avg_price_per_room"] / df["total_guests"]

        df["weekend_flag"] = (df["no_of_weekend_nights"] > 0).astype(int)

        df["previous_cancellation_ratio"] = df["no_of_previous_cancellations"] / (
            df["no_of_previous_bookings_not_canceled"] +
            df["no_of_previous_cancellations"] + 1
        )

        df["is_family"] = (df["no_of_children"] > 0).astype(int)

        logger.info("Feature engineering completed")
        return df

    except Exception as e:
        logger.exception("Error during feature engineering")
        raise e
