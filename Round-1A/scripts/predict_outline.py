import os
import json
import joblib
import pandas as pd
from scripts.extract_features import extract_layout_features


def predict_outline(pdf_path, output_dir):
    # Load model and label encoder
    clf = joblib.load("models/layout_model.pkl")
    le = joblib.load("models/label_encoder.pkl")

    # Extract features
    df = extract_layout_features(pdf_path)

    features = [
        "font_size", "len_chars", "is_upper", "page_num",
        "font_count", "is_bold", "x0", "x1", "y0", "y1"
    ]
    X = df[features]
    df["predicted"] = le.inverse_transform(clf.predict(X))

    # Prepare clean output structure
    output = {
        "title": "",
        "outline": []
    }

    # Find the first 'title' prediction
    title_row = df[df["predicted"] == "title"]
    if not title_row.empty:
        output["title"] = title_row.iloc[0]["text"]

    # Filter only H1 and H2 predictions
    headings = df[df["predicted"].isin(["H1", "H2"])]
    headings = headings.sort_values(by=["page_num", "y0"], ascending=[True, False])

    for _, row in headings.iterrows():
        output["outline"].append({
            "level": row["predicted"],
            "text": row["text"],
            "page": row["page_num"]
        })

    # Save the JSON
    filename = os.path.splitext(os.path.basename(pdf_path))[0] + ".json"
    output_path = os.path.join(output_dir, filename)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4)

    print(f" Saved outline to: {output_path}")
