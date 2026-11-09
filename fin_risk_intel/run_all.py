"""Run the full pipeline once (regenerates data, models, indexes and all artifacts)."""
from src.pipeline import get_state
import pandas as pd
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 20)

if __name__ == "__main__":
    X = get_state(force=True)
    print("\n=== Detection (test period) ===\n", X["det"]["metrics"].round(3).to_string(index=False))
    print("\n=== Hypotheses ===\n", X["hyp"][["id", "result", "verdict"]].to_string(index=False))
    print("\nArtifacts written to ./artifacts and ./data. Launch the UI with:  streamlit run app.py")
