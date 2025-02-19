import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import re
import textwrap

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

# ---------------------------
# New Styling Function for Heatmap (Ranking Table)
# ---------------------------
def color_cells_dynamic(row):
    styles = []
    try:
        # Get the reversed "vanimo" colormap.
        cmap = matplotlib.cm.get_cmap("vanimo_r")
    except Exception:
        # Fallback to reversed "viridis" if "vanimo" is not available.
        cmap = matplotlib.cm.get_cmap("viridis_r")
    for col in row.index:
        if col == "Ranking":
            styles.append("")
        else:
            cell = row[col]
            if pd.isna(cell):
                styles.append("")
            else:
                cell_str = str(cell).strip()
                if cell_str.startswith("—"):
                    # Treat missing rank as worst (ratio = 1)
                    ratio = 1.0
                else:
                    try:
                        rank_str, total_str = cell_str.split("/")
                        rank_val = float(rank_str.strip())
                        total_val = float(total_str.strip())
                        ratio = rank_val / total_val if total_val > 0 else 1.0
                    except Exception:
                        ratio = 1.0
                # Clamp ratio between 0 and 1
                ratio = max(0, min(1, ratio))
                # Get the hex color from the reversed colormap
                hex_color = matplotlib.colors.rgb2hex(cmap(ratio))
                styles.append(f"background-color: {hex_color}")
    return styles

# ---------------------------
# Session State Setup
# ---------------------------
if "matches" not in st.session_state:
    st.session_state.matches = []  # List of (label, (Institution, Country_Code)) tuples

# ---------------------------
# Streamlit Layout
# ---------------------------
st.title("Bench:red[Up]")
st.header("On your mark... bench!")

# Search Box
search_str = st.text_input("Enter partial institution name", placeholder="Enter partial institution name")

if st.button("Find Matches"):
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

# If matches are found, show a dropdown (selectbox)
if st.session_state.matches:
    match_labels = [m[0] for m in st.session_state.matches]
    selected_label = st.selectbox("Select Institution", match_labels)
    
    # Get the corresponding (Institution, Country_Code) tuple
    selected_tuple = next((tup for label, tup in st.session_state.matches if label == selected_label), None)
    
    if st.button("Display Results"):
        if not selected_tuple:
            st.error("No institution selected.")
        else:
            institution_name, country_code = selected_tuple
            # Filter the master dataset
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
                avg_totals = {}
                for ranking, group in totals.groupby("name of the ranking"):
                    avg_totals[ranking] = round(group["total"].mean())
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

                # ---------------------------
                # Apply the Dynamic Color Styling (Heatmap)
                # ---------------------------
                styled_df = result_df.style.apply(color_cells_dynamic, axis=1).hide(axis="index")
                st.markdown(styled_df.to_html(), unsafe_allow_html=True)
                
                total_appearances = sum(summary_counts.values())
                summary_parts = [f"{summary_counts[year]} in {year}" for year in years]
                st.markdown(
                    f"<br><b>{institution_name} ({country_code}) appears {total_appearances} times in total: </b> "
                    + ", ".join(summary_parts) + "<b>.</b></br>",
                    unsafe_allow_html=True
                )
                
                # ---------------------------
                # Additional Enrichment and Histograms
                # ---------------------------
                df_filtered = df_enriched[
                    (df_enriched["Institution"] == institution_name) &
                    (df_enriched["Scimago_country_code"] == country_code)
                ]
                if not df_filtered.empty:
                    record = df_filtered.iloc[0]
                    total_pubs = record.get("Total_Publications", "no match")
                    fields_str = record.get("fields", "")
                    subfields_str = record.get("Top_30_Subfields", "")
                    sdg_str = record.get("SDG", "")
                    topics_str = record.get("Top_50_Topics", "")

                    try:
                        total_pubs_int = int(total_pubs)
                    except Exception:
                        total_pubs_int = None

                    st.markdown(f"<br><b>Total Publications (2015-2024): {total_pubs}</b></br><br></br>",
                                unsafe_allow_html=True)

                    # Process fields, subfields, and SDGs data
                    fields_data = parse_topics_string(fields_str)
                    subfields_data = parse_topics_string(subfields_str)
                    if fields_data and total_pubs_int:
                        fields_data = [(name.strip(), count, count/total_pubs_int*100) 
                                       for name, count in fields_data if (count/total_pubs_int*100) > 5]
                        fields_data = sorted(fields_data, key=lambda x: x[2], reverse=True)
                    else:
                        fields_data = []
                    if subfields_data and total_pubs_int:
                        subfields_data = [(name.strip(), count, count/total_pubs_int*100) 
                                          for name, count in subfields_data if (count/total_pubs_int*100) > 3]
                        subfields_data = sorted(subfields_data, key=lambda x: x[2], reverse=True)
                    else:
                        subfields_data = []
                    sdg_data = parse_topics_string(sdg_str)
                    if sdg_data and total_pubs_int:
                        sdg_data = [(name.strip(), count, count/total_pubs_int*100)
                                    for name, count in sdg_data if (count/total_pubs_int*100) > 3]
                        sdg_data = sorted(sdg_data, key=lambda x: x[2], reverse=True)
                    else:
                        sdg_data = []
                    sdg_data_labeled = []
                    for name, count, perc in sdg_data:
                        key = name.lower().replace(",", "").replace("’", "'")
                        number = sdg_numbers.get(key, "?")
                        new_label = f"{name} (SDG {number})"
                        sdg_data_labeled.append((new_label, count, perc))
                    
                    # ---------------------------
                    # Histogram: Top Fields
                    # ---------------------------
                    if fields_data:
                        fig_fields, ax_fields = plt.subplots(figsize=(8, 6))
                        names_fields = [x[0] for x in fields_data]
                        percentages_fields = [x[2] for x in fields_data]
                        bars = ax_fields.barh(names_fields, percentages_fields, color='skyblue')
                        ax_fields.set_xlabel("Percentage (%)", fontsize=14)
                        ax_fields.set_title("Top Fields (>5%)", fontsize=16, weight="semibold")
                        ax_fields.invert_yaxis()
                        for bar, (_, count, _) in zip(bars, fields_data):
                            ax_fields.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, f"{count}", 
                                           va='center', fontsize=12)
                        ax_fields.set_yticks(range(len(names_fields)))
                        ax_fields.set_yticklabels(["\n".join(textwrap.wrap(label, width=20)) for label in names_fields])
                        ax_fields.tick_params(axis='both', labelsize=12)
                        st.pyplot(fig_fields)
                        plt.close(fig_fields)
                    else:
                        st.info("No fields data >5%.")
                    
                    # ---------------------------
                    # Histogram: Top Subfields
                    # ---------------------------
                    if subfields_data:
                        fig_subfields, ax_subfields = plt.subplots(figsize=(8, 6))
                        names_subfields = [x[0] for x in subfields_data]
                        percentages_subfields = [x[2] for x in subfields_data]
                        bars = ax_subfields.barh(names_subfields, percentages_subfields, color='lightpink')
                        ax_subfields.set_xlabel("Percentage (%)", fontsize=14)
                        ax_subfields.set_title("Top Subfields (>3%)", fontsize=16, weight="semibold")
                        ax_subfields.invert_yaxis()
                        for bar, (_, count, _) in zip(bars, subfields_data):
                            ax_subfields.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, f"{count}", 
                                              va='center', fontsize=12)
                        ax_subfields.set_yticks(range(len(names_subfields)))
                        ax_subfields.set_yticklabels(["\n".join(textwrap.wrap(label, width=30)) for label in names_subfields])
                        ax_subfields.tick_params(axis='both', labelsize=10)
                        st.pyplot(fig_subfields)
                        plt.close(fig_subfields)
                    else:
                        st.info("No subfields data >3%.")
                    
                    # ---------------------------
                    # Histogram: Top SDGs
                    # ---------------------------
                    if sdg_data_labeled:
                        fig_sdgs, ax_sdgs = plt.subplots(figsize=(8, 6))
                        names_sdgs = [x[0] for x in sdg_data_labeled]
                        percentages_sdgs = [x[2] for x in sdg_data_labeled]
                        bars = ax_sdgs.barh(names_sdgs, percentages_sdgs, color='#E6CCFF')
                        ax_sdgs.set_xlabel("Percentage (%)", fontsize=14)
                        ax_sdgs.set_title("Top SDGs (>3%)", fontsize=16, weight="semibold")
                        ax_sdgs.invert_yaxis()
                        for bar, (_, count, _) in zip(bars, sdg_data_labeled):
                            ax_sdgs.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, f"{count}", 
                                         va='center', fontsize=12)
                        ax_sdgs.set_yticks(range(len(names_sdgs)))
                        ax_sdgs.set_yticklabels(["\n".join(textwrap.wrap(label, width=20)) for label in names_sdgs])
                        ax_sdgs.tick_params(axis='both', labelsize=12)
                        st.pyplot(fig_sdgs)
                        plt.close(fig_sdgs)
                    else:
                        st.info("No SDGs data >3%.")
                    
                    # ---------------------------
                    # Topics Data: Classic Top 50 Table
                    # ---------------------------
                    topics_data = parse_topics_string(topics_str)
                    if topics_data and total_pubs_int:
                        topics_data = [(name.strip(), count, round(count/total_pubs_int*100, 2))
                                       for name, count in topics_data]
                        topics_df = pd.DataFrame(topics_data, columns=["Topic", "Count", "Ratio"])
                        # Rank topics by decreasing publication count
                        topics_df = topics_df.sort_values(by="Count", ascending=False).reset_index(drop=True)
                        topics_df = topics_df.head(50)
                        topics_df.insert(0, "Rank", range(1, len(topics_df)+1))
                        custom_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("white_yellow", ["#FFFFFF", "#FFFF99"])
                        max_ratio = topics_df["Ratio"].max()
                        styled_topics_df = topics_df.style.format({"Ratio": "{:.2f} %"}).background_gradient(
                            subset=["Ratio"], cmap=custom_cmap, vmin=0, vmax=max_ratio)
                        st.markdown(styled_topics_df.to_html(), unsafe_allow_html=True)
                    else:
                        st.info("No topics data available.")
                else:
                    st.error("No enriched record found for the selected institution.")
