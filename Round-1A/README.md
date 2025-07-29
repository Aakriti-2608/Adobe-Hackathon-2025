# PDF Outline Extractor

## Description
This tool extracts the title and headings (H1, H2, H3) from a PDF and returns a structured JSON format.

## Folder Structure

- input/: Place your input PDF files here.
- output/: Output JSON will be saved here.
- model/: Trained model and label encoder.
- scripts/: Contains feature extraction, training, and prediction scripts.
- labeled_data.csv: The training data file you label manually.

## How to Use

### Step 1: Install Requirements
```bash
pip install -r requirements.txt
```

### Step 2: Extract features from PDF
```python
from scripts.extract_features import extract_lines_with_features
df = extract_lines_with_features("input/yourfile.pdf")
df["label"] = "body"
df.to_csv("labeled_data.csv", index=False)
```

### Step 3: Manually update `labeled_data.csv` with proper labels like 'title', 'H1', 'H2', etc.

### Step 4: Train the model
```bash
python scripts/train_model.py
```

### Step 5: Run the model
```bash
python run.py
```
