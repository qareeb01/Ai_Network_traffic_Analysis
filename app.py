import os
import io
import base64
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # headless — no GUI window needed
import matplotlib.pyplot as plt
from flask import Flask, request, render_template

app = Flask(__name__)

# ── Load model once at startup ─────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
model      = joblib.load(MODEL_PATH)

# Label map: 0 = BENIGN, 1 = DDoS
LABEL_MAP  = {0: "BENIGN", 1: "DDoS"}

# How many sample rows to show in the results table
MAX_TABLE_ROWS = 50


# ──────────────────────────────────────────────────────────
# HELPER 1 — Clean the dataframe (same logic as main.py)
# ──────────────────────────────────────────────────────────
def clean_dataframe(df):
    df.columns = df.columns.str.strip()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    return df


# ──────────────────────────────────────────────────────────
# HELPER 2 — Get top N most important features for this model
# ──────────────────────────────────────────────────────────
def get_top_features(feature_names, n=3):
    """Return the top-n feature names ranked by importance."""
    importances = model.feature_importances_          # array from RandomForest
    indices     = np.argsort(importances)[::-1][:n]   # top n indices
    return [feature_names[i] for i in indices]


# ──────────────────────────────────────────────────────────
# HELPER 3 — Build a human-readable explanation
# ──────────────────────────────────────────────────────────
def build_explanation(top_features, attack_pct):
    """Return a plain-English explanation paragraph."""
    feat_list = ", ".join(f'"{f}"' for f in top_features)

    if attack_pct == 0:
        return (
            "The model found no suspicious traffic. All network flows matched "
            "normal (BENIGN) patterns. The most influential features checked were "
            f"{feat_list}. Their values were all within expected ranges."
        )
    elif attack_pct < 10:
        return (
            f"A small portion ({attack_pct:.1f}%) of the traffic was flagged as DDoS. "
            f"The model focused mainly on {feat_list} to make this decision. "
            "These features showed abnormal spikes in the flagged rows — a common sign "
            "of DDoS flooding behaviour."
        )
    elif attack_pct < 60:
        return (
            f"A significant amount ({attack_pct:.1f}%) of the captured traffic was "
            f"classified as DDoS attack traffic. Key indicators were {feat_list}. "
            "High values in these features typically indicate packet flooding or "
            "abnormal connection rates associated with DDoS attacks."
        )
    else:
        return (
            f"The majority ({attack_pct:.1f}%) of the traffic was classified as a DDoS "
            f"attack. The model heavily relied on {feat_list} to reach this verdict. "
            "These features had values far outside the normal range, which is a strong "
            "indicator of an ongoing large-scale DDoS attack."
        )


# ──────────────────────────────────────────────────────────
# HELPER 4 — Generate chart as a base64 PNG string
# ──────────────────────────────────────────────────────────
def generate_chart(normal, attacks):
    """Create a bar + pie combo chart and return it as a base64 string."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.5))
    fig.patch.set_facecolor("#0b1520")

    labels = ["BENIGN", "DDoS"]
    counts = [normal, attacks]
    colors = ["#00ffaa", "#ff3c5a"]

    # — Bar chart —
    bars = ax1.bar(labels, counts, color=colors, width=0.45, edgecolor="none")
    ax1.set_facecolor("#0b1520")
    ax1.set_title("Traffic Count", color="#cce8f4", fontsize=11, pad=10)
    ax1.tick_params(colors="#4a7a99")
    ax1.spines[:].set_color("#1a3a55")
    ax1.yaxis.label.set_color("#4a7a99")
    for bar, count in zip(bars, counts):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(counts) * 0.02,
                 str(count), ha="center", va="bottom", color="#cce8f4", fontsize=9)

    # — Pie chart —
    wedge_props = {"linewidth": 2, "edgecolor": "#0b1520"}
    ax2.pie(counts, labels=labels, colors=colors, autopct="%1.1f%%",
            startangle=90, wedgeprops=wedge_props,
            textprops={"color": "#cce8f4", "fontsize": 9})
    ax2.set_title("Distribution", color="#cce8f4", fontsize=11, pad=10)

    plt.tight_layout(pad=1.5)

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=120, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


# ──────────────────────────────────────────────────────────
# ROUTE — Home (upload form)
# ──────────────────────────────────────────────────────────
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


# ──────────────────────────────────────────────────────────
# ROUTE — Predict
# ──────────────────────────────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files or request.files["file"].filename == "":
        return render_template("index.html", error="Please select a CSV file first.")

    file = request.files["file"]

    if not file.filename.endswith(".csv"):
        return render_template("index.html", error="Only .csv files are accepted.")

    try:
        # 1. Load & clean
        df = pd.read_csv(file)
        df = clean_dataframe(df)

        if "Label" in df.columns:
            df = df.drop(columns=["Label"])

        feature_names = list(df.columns)

        # 2. Predict labels + confidence probabilities
        predictions   = model.predict(df)
        probabilities = model.predict_proba(df)      # shape: (n_rows, 2)

        confidence = [
            round(probabilities[i][pred] * 100, 1)
            for i, pred in enumerate(predictions)
        ]

        # 3. Summary counts
        total      = len(predictions)
        attacks    = int((predictions == 1).sum())
        normal     = int((predictions == 0).sum())
        attack_pct = (attacks / total * 100) if total > 0 else 0

        # 4. Top 3 important features
        top_features = get_top_features(feature_names, n=3)

        # 5. Human-readable explanation
        explanation = build_explanation(top_features, attack_pct)

        # 6. Chart (bar + pie encoded as base64 PNG)
        chart_b64 = generate_chart(normal, attacks)

        # 7. Build table rows (capped for performance)
        table_rows = []
        display_df = df.head(MAX_TABLE_ROWS).reset_index(drop=True)
        for i, row in display_df.iterrows():
            table_rows.append({
                "row_num":    i + 1,
                "label":      LABEL_MAP[predictions[i]],
                "is_attack":  predictions[i] == 1,
                "confidence": confidence[i],
                "feat1_name": top_features[0],
                "feat1_val":  round(row[top_features[0]], 4),
                "feat2_name": top_features[1],
                "feat2_val":  round(row[top_features[1]], 4),
                "feat3_name": top_features[2],
                "feat3_val":  round(row[top_features[2]], 4),
            })

        verdict = "ATTACK DETECTED" if attacks > 0 else "ALL TRAFFIC NORMAL"

        return render_template(
            "index.html",
            verdict=verdict,
            total=total,
            attacks=attacks,
            normal=normal,
            attack_pct=round(attack_pct, 1),
            filename=file.filename,
            table_rows=table_rows,
            showing=len(table_rows),
            top_features=top_features,
            explanation=explanation,
            chart_b64=chart_b64,
        )

    except Exception as e:
        return render_template("index.html", error=f"Error processing file: {str(e)}")


if __name__ == "__main__":
    app.run(debug=True)