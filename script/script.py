import sys

from azureml.core import Run
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


def main(
    train_feature_path,
    train_label_path,
    test_feature_path,
    submission_format_path,
    submission_path,
    model_path,
):
    run = Run.get_context()
    features_df = pd.read_csv(train_feature_path, index_col="respondent_id")
    labels_df = pd.read_csv(train_label_path, index_col="respondent_id")
    numeric_cols = features_df.columns[features_df.dtypes != "object"].values

    # chain preprocessing into a Pipeline object
    numeric_preprocessing_steps = Pipeline(
        [
            ("standard_scaler", StandardScaler()),
            ("simple_imputer", SimpleImputer(strategy="median")),
        ]
    )

    # create the preprocessor stage of final pipeline
    preprocessor = ColumnTransformer(
        transformers=[("numeric", numeric_preprocessing_steps, numeric_cols)], remainder="drop"
    )

    estimators = MultiOutputClassifier(estimator=LogisticRegression(penalty="l2", C=1))

    full_pipeline = Pipeline([("preprocessor", preprocessor), ("estimators", estimators)])

    X_train, X_eval, y_train, y_eval = train_test_split(
        features_df,
        labels_df,
        test_size=0.33,
        shuffle=True,
        stratify=labels_df,
    )

    full_pipeline.fit(X_train, y_train)
    preds = full_pipeline.predict_proba(X_eval)

    y_preds = pd.DataFrame(
        {"h1n1_vaccine": preds[0][:, 1], "seasonal_vaccine": preds[1][:, 1]}, index=y_eval.index
    )
    run.log("validationRocAuc", roc_auc_score(y_eval, y_preds))

    full_pipeline.fit(features_df, labels_df)

    test_features_df = pd.read_csv(test_feature_path, index_col="respondent_id")
    test_probas = full_pipeline.predict_proba(test_features_df)
    submission_df = pd.read_csv(submission_format_path, index_col="respondent_id")

    # Make sure we have the rows in the same order
    np.testing.assert_array_equal(test_features_df.index.values, submission_df.index.values)

    # Save predictions to submission data frame
    submission_df["h1n1_vaccine"] = test_probas[0][:, 1]
    submission_df["seasonal_vaccine"] = test_probas[1][:, 1]

    submission_df.to_csv(submission_path, index=True)


if __name__ == "__main__":
    train_features_path = sys.argv[1]
    train_labels_path = sys.argv[2]
    test_features_path = sys.argv[3]
    submission_format_path = sys.argv[4]
    submission_path = sys.argv[5]
    model_path = sys.argv[6]
    main(
        train_features_path,
        train_labels_path,
        test_features_path,
        submission_format_path,
        submission_path,
        model_path,
    )
