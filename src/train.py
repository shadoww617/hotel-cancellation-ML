import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

from xgboost import XGBClassifier

from logger import get_logger

logger = get_logger(__name__)


def train_models(X, y):
    try:
        logger.info("Starting model training with XGBoost")

        num_features = X.select_dtypes(include=np.number).columns.tolist()
        cat_features = X.select_dtypes(exclude=np.number).columns.tolist()

        preprocessor = ColumnTransformer([
            ("num", "passthrough", num_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)
        ])

        models = {
            "logreg": Pipeline([
                ("pre", preprocessor),
                ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))
            ]),

            "rf": Pipeline([
                ("pre", preprocessor),
                ("smote", SMOTE(random_state=42)),
                ("clf", RandomForestClassifier(random_state=42))
            ]),

            "xgboost": Pipeline([
                ("pre", preprocessor),
                ("smote", SMOTE(random_state=42)),
                ("clf", XGBClassifier(
                    objective="binary:logistic",
                    eval_metric="logloss",
                    use_label_encoder=False,
                    random_state=42,
                    scale_pos_weight=(y.value_counts()[0] / y.value_counts()[1])
                ))
            ])
        }

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        # 1. TRAIN BASELINE MODELS
        for name, model in models.items():
            logger.info(f"Training model: {name}")
            model.fit(X_train, y_train)

        # 2. HYPERPARAMETER TUNING (XGBOOST)
        logger.info("Starting XGBoost hyperparameter tuning")

        param_dist = {
            "clf__n_estimators": [200, 300, 500],
            "clf__max_depth": [3, 5, 8],
            "clf__learning_rate": [0.01, 0.05, 0.1],
            "clf__subsample": [0.8, 1.0],
            "clf__colsample_bytree": [0.8, 1.0]
        }

        search = RandomizedSearchCV(
            models["xgboost"],
            param_distributions=param_dist,
            n_iter=10,
            scoring="roc_auc",
            cv=3,
            verbose=2,
            n_jobs=-1
        )

        search.fit(X_train, y_train)
        best_model = search.best_estimator_

        # 3. SAVE BEST MODEL
        os.makedirs("models", exist_ok=True)
        joblib.dump(best_model, "models/best_model.joblib")

        logger.info("Best XGBoost model saved to models/best_model.joblib")

        return best_model, X_test, y_test

    except Exception as e:
        logger.exception("Error during XGBoost training")
        raise e
