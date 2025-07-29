from scripts.predict_outline import predict_outline
import os

def main():
    input_dir = "input"
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(input_dir, filename)
            print(f"🔍 Processing: {filename}")
            predict_outline(pdf_path, output_dir)

if __name__ == "__main__":
    main()
