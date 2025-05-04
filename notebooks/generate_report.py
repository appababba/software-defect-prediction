# generate_report.py

import pandas as pd
import glob

def create_html_report():
    # Read the summary metrics
    summary = pd.read_csv("summary_metrics.csv")

    # Begin HTML
    html = [
        "<!DOCTYPE html>",
        "<html><head>",
        "<meta charset='utf-8'>",
        "<title>Defect Prediction Report</title>",
        "</head><body>",
        "<h1>Defect Prediction Summary</h1>",
        "<h2>Summary Metrics</h2>",
        summary.to_html(index=False, classes='table table-striped'),
    ]

    # Sections of plots
    sections = [
        ("Confusion Matrices", "*_confusion.png"),
        ("ROC Curves",         "*_roc.png"),
        ("Precision-Recall Curves", "*_pr.png"),
    ]

    for title, pattern in sections:
        html.append(f"<h2>{title}</h2>")
        for fn in sorted(glob.glob(pattern)):
            html.append(f"<h4>{fn}</h4>")
            html.append(f'<img src="{fn}" style="max-width:800px;margin-bottom:20px;">')

    # End HTML
    html.append("</body></html>")

    with open("report.html", "w", encoding="utf-8") as f:
        f.write("\n".join(html))

if __name__ == "__main__":
    create_html_report()
    print("Generated report.html")
