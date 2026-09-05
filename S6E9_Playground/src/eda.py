"""Quick EDA report -> outputs/eda_report.txt + a few PNGs."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .config import OUTPUT_DIR, TARGET
from .data import load_train

sns.set_theme(style="whitegrid")


def main():
    df = load_train()
    out = OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append(f"shape: {df.shape}")
    lines.append(f"target rate:\n{(df[TARGET].value_counts(normalize=True)).to_string()}")
    lines.append(f"\nmissing:\n{df.isna().sum().to_string()}")
    lines.append(f"\ndescribe:\n{df.describe(include='all').T.to_string()}")

    y = (df[TARGET] == "Yes").astype(int)
    num_cols = df.select_dtypes(include="number").columns.drop("id")
    corr = df[num_cols].corrwith(y).sort_values(ascending=False)
    lines.append(f"\ncorr with target:\n{corr.to_string()}")

    for c in ["Subsidy_Available", "Range_Anxiety_Level", "Home_Charging_Possible",
              "City_Type", "Current_Car_Type", "Gender"]:
        ct = pd.crosstab(df[c], df[TARGET], normalize="index").round(4)
        lines.append(f"\n=== {c} ===\n{ct.to_string()}")

    (out / "eda_report.txt").write_text("\n".join(lines))
    print("\n".join(lines[:30]))

    # Plots
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.countplot(data=df, x=TARGET, ax=axes[0])
    axes[0].set_title("Target distribution")
    corr.plot(kind="bar", ax=axes[1])
    axes[1].set_title("Numeric corr with Will_Buy_EV=Yes")
    fig.tight_layout()
    fig.savefig(out / "eda_target_corr.png", dpi=120)
    print(f"saved -> {out / 'eda_report.txt'}, {out / 'eda_target_corr.png'}")


if __name__ == "__main__":
    main()
