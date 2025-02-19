import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.ticker as mticker
import re
import textwrap

# Set page config to use full width
st.set_page_config(
    page_title="Institution Analysis Dashboard",
    layout="wide"
)

# ---------------------------
# Caching Data Loading
# ---------------------------
@st.cache_data
def load_data():
    df_master = pd.read_csv("data/Scimago_results_2021_2024.csv", low_memory=False)
    df_enriched = pd.read_csv("data/unique_institutions_enriched.csv", encoding="utf-8-sig")
    return df_master, df_enriched

df_master, df_enriched = load_data()

# ---------------------------
# SDG Official Numbers Dictionary
# ---------------------------
sdg_numbers = {
    "no poverty": 1,
    "zero hunger": 2,
    "good health and well-being": 3,
    "quality education": 4,
    "gender equality": 5,
    "clean water and sanitation": 6,
    "affordable and clean energy": 7,
    "decent work and economic growth": 8,
    "industry, innovation and infrastructure": 9,
    "reduced inequalities": 10,
    "sustainable cities and communities": 11,
    "responsible consumption and production": 12,
    "climate action": 13,
    "life below water": 14,
    "life on land": 15,
    "peace, justice and strong institutions": 16,
    "partnerships for the goals": 17
}
# Additional variants (if keys from data differ)
sdg_variants = {
    "peace and strong institution": 16,
    "industry and innovations": 9,
}

# ---------------------------
# Helper Functions
# ---------------------------
def parse_topics_string(s):
    if pd.isna(s) or s.strip() == "":
        return []
    pattern = r'(.+?)\s*\((\d+)\)(?:;\s*|$)'
    matches = re.findall(pattern, s)
    return [(match[0].strip(), int(match[1])) for match in matches]

def truncate_text(x, max_chars=100):
    if isinstance(x, str) and len(x) > max_chars:
        return x[:max_chars] + "..."
    return x

def get_heatmap_color(ratio):
    """
    Given a ratio between 0 and 1, interpolate between:
      - 0: #ef476f (red)
      - 0.5: #ffd166 (yellow)
      - 1: #06d6a0 (green)
    """
    c0 = (239/255, 71/255, 111/255)    # red
    c_mid = (255/255, 209/255, 102/255)  # yellow
    c1 = (6/255, 214/255, 160/255)       # green

    if ratio <= 0.5:
        w = ratio / 0.5
        r_val = (1 - w)*c0[0] + w*c_mid[0]
        g_val = (1 - w)*c0[1] + w*c_mid[1]
        b_val = (1 - w)*c0[2] + w*c_mid[2]
    else:
        w = (ratio - 0.5) / 0.5
        r_val = (1 - w)*c_mid[0] + w*c1[0]
        g_val = (1 - w)*c_mid[1] + w*c1[1]
        b_val = (1 - w)*c_mid[2] + w*c1[2]
    return matplotlib.colors.rgb2hex((r_val, g_val, b_val))

def fix_width(cell, width=11):
    """Return a string forced to a fixed width (left-justified)."""
    s = str(cell)
    if len(s) > width:
        return s[:width]
    return s.ljust(width)

# ---------------------------
# New Styling Function for Heatmap (Ranking Table)
# ---------------------------
def color_cells_dynamic(row):
    styles = []
    for col in row.index:
        if col == "Ranking":
            styles.append("")  # Leave Ranking column unmodified.
        else:
            cell = row[col]
            # Check for missing data (NaN or "no data")
            if pd.isna(cell) or str(cell).strip().lower() == "no data":
                # Force white background, black text, and smaller font for missing data.
                styles.append("background-color: white; color: black; font-size: 10px;")
            else:
                cell_str = str(cell).strip()
                if cell_str.startswith("—"):
                    ratio = 0.999
                else:
                    try:
                        rank_str, total_str = cell_str.split("/")
                        rank_val = float(rank_str.strip())
                        total_val = float(total_str.strip())
                        ratio = rank_val / total_val if total_val > 0 else 0.0
                    except Exception:
                        ratio = 0.0
                ratio = max(0, min(1, ratio))
                # Reverse ratio for desired color order
                hex_color = get_heatmap_color(1 - ratio)
                styles.append(f"background-color: {hex_color}; color: black;")
    return styles

# ---------------------------
# Session State Setup
# ---------------------------
if "matches" not in st.session_state:
    st.session_state.matches = []  # List of (label, (Institution, Country_Code)) tuples

# ---------------------------
# Main Layout: Search and Results in Columns
# ---------------------------
st.title("Bench:red[Up]")
st.header("On your mark... bench!")

col1, col2 = st.columns([1, 3])
with col1:
    search_str = st.text_input("Enter partial institution name", placeholder="Enter partial institution name")
with col2:
    find_button = st.button("Find Matches")

if find_button:
    search_str = search_str.strip().lower()
    if not search_str:
        st.warning("Please enter a non-empty search string.")
        st.session_state.matches = []
    else:
        matches = [
            (f"{row['Institution']} ({row['Scimago_country_code']})",
             (row['Institution'], row['Scimago_country_code']))
            for _, row in df_enriched.iterrows()
            if search_str in row['Institution'].lower()
        ]
        if not matches:
            st.info(f"No institutions found containing '{search_str}'.")
            st.session_state.matches = []
        else:
            st.success(f"Found {len(matches)} match(es). Please select one below.")
            st.session_state.matches = matches

if st.session_state.matches:
    selected_label = st.selectbox("Select Institution", [m[0] for m in st.session_state.matches])
    selected_tuple = next((tup for label, tup in st.session_state.matches if label == selected_label), None)
    
    if st.button("Display Results"):
        if not selected_tuple:
            st.error("No institution selected.")
        else:
            institution_name, country_code = selected_tuple
            df_inst = df_master[
                (df_master["institution"] == institution_name) &
                (df_master["country"] == country_code)
            ]
            if df_inst.empty:
                st.error(f"No ranking data found for {institution_name} ({country_code}).")
            else:
                # ---------------------------
                # Processing Ranking Data
                # ---------------------------
                totals = df_master.groupby(["name of the ranking", "year"]).size().reset_index(name="total")
                total_dict = {(row["name of the ranking"], row["year"]): row["total"]
                              for _, row in totals.iterrows()}
                inst_dict = {(row["name of the ranking"], row["year"]): row["rank"]
                             for _, row in df_inst.iterrows()}
                years = [2021, 2022, 2023, 2024]
                ranking_names = sorted(df_master["name of the ranking"].unique(),
                                       key=lambda x: (0 if x.lower() == "all subject areas" else 1, x))
                output_data = {}
                summary_counts = {year: 0 for year in years}
                for ranking in ranking_names:
                    if not any((ranking, year) in inst_dict for year in years):
                        continue
                    row_vals = {}
                    for year in years:
                        key = (ranking, year)
                        if key not in total_dict:
                            row_vals[year] = np.nan
                        else:
                            total = total_dict[key]
                            if key in inst_dict:
                                cell_val = f"{inst_dict[key]} / {total}"
                                summary_counts[year] += 1
                            else:
                                cell_val = f"— / {total}"
                            row_vals[year] = cell_val
                    output_data[ranking] = row_vals
                result_df = pd.DataFrame.from_dict(output_data, orient="index", columns=years)
                result_df.index.name = "Ranking"
                result_df = result_df.reset_index()
                
                # Replace NaN with "no data"
                result_df = result_df.fillna("no data")
                
                # Force fixed width only on the result columns (11 characters)
                for col in years:
                    result_df[col] = result_df[col].apply(lambda x: fix_width(x, 11))
                
                # ---------------------------
                # Display Scimago Results Header and Heatmap
                # ---------------------------
                st.markdown("<h3>Scimago results</h3>", unsafe_allow_html=True)
                st.markdown("Thematic rankings with no data started in 2022.", unsafe_allow_html=True)
                styled_df = result_df.style.apply(color_cells_dynamic, axis=1).hide(axis="index")
                st.markdown(styled_df.to_html(), unsafe_allow_html=True)
                
                total_appearances = sum(summary_counts.values())
                summary_parts = [f"{summary_counts[year]} in {year}" for year in years]
                st.markdown(
                    f"{institution_name} ({country_code}) appears {total_appearances} times in total: " +
                    ", ".join(summary_parts) + ".",
                    unsafe_allow_html=True
                )
                
                # ---------------------------
                # Display OpenAlex Results Header and Total Publications
                # ---------------------------
                st.markdown("<h3>OpenAlex results</h3>", unsafe_allow_html=True)
                try:
                    total_pubs_int = int(df_enriched[
                        (df_enriched["Institution"] == institution_name) &
                        (df_enriched["Scimago_country_code"] == country_code)
                    ].iloc[0].get("Total_Publications", "no match"))
                    total_pubs_str = f"{total_pubs_int:,}"
                except Exception:
                    total_pubs_str = "no match"
                st.markdown(f"<b>Total publications (articles only) for the period 2015-2024: <span style='color:red'>{total_pubs_str}</span></b>",
                            unsafe_allow_html=True)
                
                # ---------------------------
                # Additional Enrichment and Histograms
                # ---------------------------
                df_filtered = df_enriched[
                    (df_enriched["Institution"] == institution_name) &
                    (df_enriched["Scimago_country_code"] == country_code)
                ]
                if not df_filtered.empty:
                    record = df_filtered.iloc[0]
                    fields_str = record.get("fields", "")
                    subfields_str = record.get("Top_30_Subfields", "")
                    sdg_str = record.get("SDG", "")
                    topics_str = record.get("Top_50_Topics", "")
                    
                    try:
                        total_pubs_int = int(record.get("Total_Publications", "0"))
                    except Exception:
                        total_pubs_int = None
                    
                    if fields_str and total_pubs_int:
                        fields_data = [(name.strip(), count, count/total_pubs_int*100) 
                                       for name, count in parse_topics_string(fields_str)
                                       if (count/total_pubs_int*100) > 5]
                        fields_data = sorted(fields_data, key=lambda x: x[2], reverse=True)
                    else:
                        fields_data = []
                    if subfields_str and total_pubs_int:
                        subfields_data = [(name.strip(), count, count/total_pubs_int*100) 
                                          for name, count in parse_topics_string(subfields_str)
                                          if (count/total_pubs_int*100) > 3]
                        subfields_data = sorted(subfields_data, key=lambda x: x[2], reverse=True)
                    else:
                        subfields_data = []
                    if sdg_str and total_pubs_int:
                        sdg_data = [(name.strip(), count, count/total_pubs_int*100)
                                    for name, count in parse_topics_string(sdg_str)
                                    if (count/total_pubs_int*100) > 1]
                        sdg_data = sorted(sdg_data, key=lambda x: x[2], reverse=True)
                    else:
                        sdg_data = []
                    sdg_data_labeled = []
                    for name, count, perc in sdg_data:
                        key = name.lower().replace(",", "").replace("’", "'")
                        if key in sdg_numbers:
                            number = sdg_numbers[key]
                        elif key in sdg_variants:
                            number = sdg_variants[key]
                        else:
                            number = "?"
                        new_label = f"{name} (SDG {number})"
                        sdg_data_labeled.append((new_label, count, perc))
                    
                    # Formatter for x-axis ticks to show integer percentages.
                    formatter = mticker.FuncFormatter(lambda x, pos: f"{int(round(x))} %")
                    
                    # ---------------------------
                    # Histogram: Top Fields
                    # ---------------------------
                    if fields_data:
                        st.subheader("Top Fields (>5%)")
                        fig_fields, ax_fields = plt.subplots(figsize=(10, 5))
                        names_fields = [x[0] for x in fields_data]
                        percentages_fields = [x[2] for x in fields_data]
                        bars = ax_fields.barh(names_fields, percentages_fields, color='#16a4d8')
                        ax_fields.set_xlabel("Percentage of 2015-2024 publications", fontsize=10)
                        ax_fields.xaxis.set_major_formatter(formatter)
                        ax_fields.invert_yaxis()
                        for bar, (_, count, _) in zip(bars, fields_data):
                            ax_fields.annotate(f"{count:,}",
                                               xy=(bar.get_width(), bar.get_y() + bar.get_height()/2),
                                               xytext=(3, 0), textcoords="offset points",
                                               va='center', fontsize=10)
                        ax_fields.margins(x=0.05)
                        st.pyplot(fig_fields)  # Removed use_container_width=True
                    else:
                        st.info("No fields data >5%.")
                    
                    # ---------------------------
                    # Histogram: Top Subfields
                    # ---------------------------
                    if subfields_data:
                        st.subheader("Top Subfields (>3%)")
                        fig_subfields, ax_subfields = plt.subplots(figsize=(10, 6))
                        names_subfields = [x[0] for x in subfields_data]
                        percentages_subfields = [x[2] for x in subfields_data]
                        bars = ax_subfields.barh(names_subfields, percentages_subfields, color='#60dbe8')
                        ax_subfields.set_xlabel("Percentage of 2015-2024 publications", fontsize=10)
                        ax_subfields.xaxis.set_major_formatter(formatter)
                        ax_subfields.invert_yaxis()
                        for bar, (_, count, _) in zip(bars, subfields_data):
                            ax_subfields.annotate(f"{count:,}",
                                                  xy=(bar.get_width(), bar.get_y() + bar.get_height()/2),
                                                  xytext=(3, 0), textcoords="offset points",
                                                  va='center', fontsize=10)
                        st.pyplot(fig_subfields)
                    else:
                        st.info("No subfields data >3%.")
                    
                    # ---------------------------
                    # Histogram: Top SDGs
                    # ---------------------------
                    if sdg_data_labeled:
                        st.subheader("Top SDGs (>1%)")
                        fig_sdgs, ax_sdgs = plt.subplots(figsize=(10, 5))
                        names_sdgs = [x[0] for x in sdg_data_labeled]
                        percentages_sdgs = [x[2] for x in sdg_data_labeled]
                        bars = ax_sdgs.barh(names_sdgs, percentages_sdgs, color='#9b5fe0')
                        ax_sdgs.set_xlabel("Percentage of 2015-2024 publications", fontsize=10)
                        ax_sdgs.xaxis.set_major_formatter(formatter)
                        ax_sdgs.invert_yaxis()
                        for bar, (_, count, _) in zip(bars, sdg_data_labeled):
                            ax_sdgs.annotate(f"{count:,}",
                                             xy=(bar.get_width(), bar.get_y() + bar.get_height()/2),
                                             xytext=(3, 0), textcoords="offset points",
                                             va='center', fontsize=10)
                        st.pyplot(fig_sdgs)
                    else:
                        st.info("No SDGs data >1%.")
                    
                    # ---------------------------
                    # Topics Data: Classic Top 50 Table
                    # ---------------------------
                    topics_data = parse_topics_string(topics_str)
                    if topics_data and total_pubs_int:
                        topics_data = [(name.strip(), count, round(count/total_pubs_int*100, 2))
                                       for name, count in topics_data]
                        topics_df = pd.DataFrame(topics_data, columns=["Topic", "Count", "Ratio"])
                        topics_df = topics_df.sort_values(by="Count", ascending=False).reset_index(drop=True)
                        topics_df = topics_df.head(50)
                        topics_df.insert(0, "Rank", range(1, len(topics_df)+1))
                        custom_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
                            "custom_yellow", ["#FFFFFF", "#d9bc2b", "#695806"]
                        )
                        styled_topics_df = topics_df.style.format({"Ratio": "{:.2f} %"}).background_gradient(
                            subset=["Ratio"], cmap=custom_cmap, vmin=0, vmax=6
                        ).hide(axis="index")
                        st.markdown(styled_topics_df.to_html(), unsafe_allow_html=True)
                    else:
                        st.info("No topics data available.")
                else:
                    st.error("No enriched record found for the selected institution.")
