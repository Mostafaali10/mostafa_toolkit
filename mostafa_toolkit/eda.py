import pandas as pd
from IPython.display import display, Markdown

from .constants import (
    DEFAULT_MAX_CARDINALITY,
    FONTSIZE_OVERVIEW_TABLE,
    FONTSIZE_STATS_TABLE,
    COLOR_RED_BAR,
    COLOR_BLUE_BAR
)

def statistical_analysis(df, max_cardinality=DEFAULT_MAX_CARDINALITY):
    """Clean statistical display with improved table formatting."""

    display(Markdown("## 📊 1. General Overview"))

    overview = pd.DataFrame({
        'Metric': ['Rows', 'Columns', 'Duplicates', 'Total Missing Values'],
        'Value': [df.shape[0], df.shape[1], df.duplicated().sum(), df.isnull().sum().sum()]
    }).set_index('Metric')
    overview.index.name = None

    display(
        overview.style
        .set_properties(**{'font-size': FONTSIZE_OVERVIEW_TABLE})
        .set_caption("Dataset Summary")
    )

    print("\n")
    display(Markdown("## 🧾 2. Data Types & Missing Values"))

    info_df = pd.DataFrame({
        'Type': df.dtypes,
        'Null Count': df.isnull().sum(),
        'Null %': (df.isnull().sum() / len(df)) * 100
    })

    styled_info = (
        info_df.sort_values(by='Null Count', ascending=False)
        .style
        .format({'Null %': "{:.2f}%"})
        .background_gradient(subset=['Null %'], cmap='Reds')
        .bar(subset=['Null Count'], color=COLOR_RED_BAR)
        .set_properties(**{'font-size': FONTSIZE_STATS_TABLE})
    )

    display(styled_info)

    print("\n")
    display(Markdown("## 📈 3. Numerical Statistics"))

    num_stats = df.describe().T

    display(
        num_stats.style
        .background_gradient(cmap='YlGnBu')
        .format("{:.3f}")
        .set_properties(**{'font-size': FONTSIZE_STATS_TABLE})
    )

    print("\n")
    display(Markdown("## 🧩 4. Categorical Value Counts"))

    cat_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()

    if not cat_columns:
        display(Markdown("*No categorical columns found.*"))
        return

    for col in cat_columns:

        n_unique = df[col].nunique()

        display(Markdown(f"### 🔹 {col}  *(Unique values: {n_unique})*"))

        if n_unique > max_cardinality:
            display(Markdown(
                f"⚠️ Skipped — High cardinality ({n_unique} unique values)"
            ))
            continue

        val_counts = df[col].value_counts().to_frame(name="Count")
        val_counts.index.name = None
        display(
            val_counts.style
            .bar(color=COLOR_BLUE_BAR, vmin=0)
            .set_properties(**{'font-size': FONTSIZE_STATS_TABLE})
        )
        print("\n")
