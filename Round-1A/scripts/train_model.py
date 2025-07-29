import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import joblib


def train_and_save_model(csv_path="labeled_data.csv"):
    df = pd.read_csv(csv_path)

    # Encode string labels to integers
    le = LabelEncoder()
    df["label_encoded"] = le.fit_transform(df["label"])

    # Features to use in the model
    features = [
        "font_size", "len_chars", "is_upper", "page_num",
        "font_count", "is_bold", "x0", "x1", "y0", "y1"
    ]

    X = df[features]
    y = df["label_encoded"]
    

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Apply SMOTE
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    # Train the model
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)

    # Predict on test set
    y_pred = clf.predict(X_test)

    # Classification report
    print(" Classification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # Precision scores
    precision_macro = precision_score(y_test, y_pred, average="macro")
    precision_weighted = precision_score(y_test, y_pred, average="weighted")
    print(f" Macro Precision: {precision_macro:.4f}")
    print(f" Weighted Precision: {precision_weighted:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
    disp.plot(xticks_rotation=45)
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()

    # Save model and label encoder
    os.makedirs("models", exist_ok=True)
    joblib.dump(clf, "models/layout_model.pkl")
    joblib.dump(le, "models/label_encoder.pkl")
    print(" Model and encoder saved.")


if __name__ == "__main__":
    train_and_save_model()