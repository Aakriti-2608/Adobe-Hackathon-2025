
import pandas as pd
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer, LTChar
import os

def extract_layout_features(pdf_path):
    data = []
    for page_num, layout in enumerate(extract_pages(pdf_path)):
        for element in layout:
            if isinstance(element, LTTextContainer):
                for text_line in element:
                    text = ""
                    font_sizes = []
                    bold_flags = []
                    positions = []
                    fonts = set()

                    if hasattr(text_line, "__iter__"):
                        for char in text_line:
                            if isinstance(char, LTChar):
                                font_sizes.append(char.size)
                                fonts.add(char.fontname)
                                bold_flags.append("Bold" in char.fontname)
                                positions.append((char.x0, char.y0, char.x1, char.y1))
                                text += char.get_text()

                    if text.strip():
                        avg_font = sum(font_sizes) / len(font_sizes) if font_sizes else 0
                        is_bold = int(any(bold_flags))
                        is_upper = int(text.strip().isupper())
                        len_chars = len(text.strip())
                        font_count = len(fonts)

                        if positions:
                            x0 = min(p[0] for p in positions)
                            y0 = min(p[1] for p in positions)
                            x1 = max(p[2] for p in positions)
                            y1 = max(p[3] for p in positions)
                        else:
                            x0 = y0 = x1 = y1 = 0

                        data.append({
                            "text": text.strip(),
                            "font_size": avg_font,
                            "len_chars": len_chars,
                            "is_upper": is_upper,
                            "page_num": page_num + 1,
                            "font_count": font_count,
                            "is_bold": is_bold,
                            "x0": x0,
                            "x1": x1,
                            "y0": y0,
                            "y1": y1
                        })

    return pd.DataFrame(data)

# Batch multiple PDFs into one dataset
def extract_from_multiple_pdfs(folder_path):
    all_dfs = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            full_path = os.path.join(folder_path, filename)
            df = extract_layout_features(full_path)
            df["source_pdf"] = filename
            all_dfs.append(df)
    return pd.concat(all_dfs, ignore_index=True)
