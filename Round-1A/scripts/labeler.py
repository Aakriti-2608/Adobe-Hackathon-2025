import os
import pandas as pd
from extract_features import extract_layout_features

INPUT_FOLDER = "input"
OUTPUT_CSV = "labeled_data.csv"

def apply_label_rules(df):
    max_font = df["font_size"].max()

    def label_row(row):
        if row["font_size"] >= max_font - 1 and row["is_bold"] == 1 and row["page_num"] == 1:
            return "title"
        elif row["is_upper"] and row["is_bold"] and row["font_size"] >= 14:
            return "H1"
        elif row["is_bold"] and row["font_size"] >= 13:
            return "H2"
        elif row["is_bold"] or (row["font_size"] > 12):
            return "H3"
        else:
            return "body"

    df["label"] = df.apply(label_row, axis=1)
    return df

# Process all PDFs
def process_all_pdfs():
    all_data = []

    for filename in os.listdir(INPUT_FOLDER):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(INPUT_FOLDER, filename)
            print(f"🔍 Extracting and labeling: {filename}")
            df = extract_layout_features(pdf_path)
            df = apply_label_rules(df)
            all_data.append(df)

    #Combine all
    final_df = pd.concat(all_data, ignore_index=True)

    #Save to CSV
    final_df.to_csv(OUTPUT_CSV, index=False)
    print(f" Auto-labeled dataset saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    process_all_pdfs()
