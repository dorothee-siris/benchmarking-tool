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
# Session State Setup
# ---------------------------
if "matches" not in st.session_state:
    st.session_state.matches = []  # List of (label, (Institution, Country_Code)) tuples

# ---------------------------
# Streamlit Layout
# ---------------------------
st.title("Institution Ranking Dashboard")

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
                # Styling the Results Table
                # ---------------------------
                def color_cells(row):
                    my_palette = [
                        "#009900",  # best (green)
                        "#339933",
                        "#66cc66",
                        "#99cc66",
                        "#ccdd66",
                        "#f2e89d",  # light yellow (middle)
                        "#f2c9a0",
                        "#f2aa93",
                        "#f28985",
                        "#f26777",
                        "#f25252"   # worst (red)
                    ]
                    missing_color = "#f03737"
                    styles = []
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
                                    styles.append(f"background-color: {missing_color}")
                                else:
                                    try:
                                        rank_str, _ = cell_str.split("/")
                                        rank_val = float(rank_str.strip())
                                    except Exception:
                                        rank_val = 0
                                    ranking = row["Ranking"]
                                    avg_total = avg_totals.get(ranking, 1)
                                    ratio = rank_val / avg_total
                                    index = int(ratio * (len(my_palette) - 1))
                                    color = my_palette[index]
                                    styles.append(f"background-color: {color}")
                    return styles

                def bold_all_subject(row):
                    return ["font-weight: bold" if (col=="Ranking" and row["Ranking"].lower() == "all subject areas") else "" 
                            for col in row.index]

                styled_df = result_df.style.apply(color_cells, axis=1)\
                                            .apply(bold_all_subject, axis=1)\
                                            .hide(axis="index")
                # Render the styled HTML table in Streamlit
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

                    if (fields_data or subfields_data or sdg_data_labeled) and total_pubs_int:
                        fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(18, 10))

                        if fields_data:
                            names_fields = [x[0] for x in fields_data]
                            percentages_fields = [x[2] for x in fields_data]
                            bars1 = ax1.barh(names_fields, percentages_fields, color='skyblue')
                            ax1.set_xlabel("Percentage (%)", fontsize=14)
                            ax1.set_title("Top Fields (>5%)", fontsize=16, weight="semibold")
                            ax1.invert_yaxis()
                            for bar, (_, count, _) in zip(bars1, fields_data):
                                ax1.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, f"{count}",
                                         va='center', fontsize=12)
                            ax1.set_yticks(range(len(names_fields)))
                            ax1.set_yticklabels(["\n".join(textwrap.wrap(label, width=20)) for label in names_fields])
                            ax1.tick_params(axis='both', labelsize=12)
                        else:
                            ax1.text(0.5, 0.5, "No fields data >5%", ha="center", va="center", fontsize=14)

                        if subfields_data:
                            names_subfields = [x[0] for x in subfields_data]
                            percentages_subfields = [x[2] for x in subfields_data]
                            bars2 = ax2.barh(names_subfields, percentages_subfields, color='lightpink')
                            ax2.set_xlabel("Percentage (%)", fontsize=14)
                            ax2.set_title("Top Subfields (>3%)", fontsize=16, weight="semibold")
                            ax2.invert_yaxis()
                            for bar, (_, count, _) in zip(bars2, subfields_data):
                                ax2.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, f"{count}",
                                         va='center', fontsize=12)
                            ax2.set_yticks(range(len(names_subfields)))
                            ax2.set_yticklabels(["\n".join(textwrap.wrap(label, width=30)) for label in names_subfields])
                            ax2.tick_params(axis='both', labelsize=10)
                        else:
                            ax2.text(0.5, 0.5, "No subfields data >3", ha="center", va="center", fontsize=14)

                        if sdg_data_labeled:
                            names_sdgs = [x[0] for x in sdg_data_labeled]
                            percentages_sdgs = [x[2] for x in sdg_data_labeled]
                            bars3 = ax3.barh(names_sdgs, percentages_sdgs, color='#E6CCFF')
                            ax3.set_xlabel("Percentage (%)", fontsize=14)
                            ax3.set_title("Top SDGs (>3%)", fontsize=16, weight="semibold")
                            ax3.invert_yaxis()
                            for bar, (_, count, _) in zip(bars3, sdg_data_labeled):
                                ax3.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, f"{count}",
                                         va='center', fontsize=12)
                            ax3.set_yticks(range(len(names_sdgs)))
                            ax3.set_yticklabels(["\n".join(textwrap.wrap(label, width=20)) for label in names_sdgs])
                            ax3.tick_params(axis='both', labelsize=12)
                        else:
                            ax3.text(0.5, 0.5, "No SDGs data >3", ha="center", va="center", fontsize=14)

                        plt.tight_layout()
                        st.pyplot(fig)
                    else:
                        st.info("No valid data for histograms.")

                    topics_data = parse_topics_string(topics_str)
                    if topics_data and total_pubs_int:
                        topics_data = [(name.strip(), count, round(count/total_pubs_int*100, 2))
                                       for name, count in topics_data]
                        topics_df = pd.DataFrame(topics_data, columns=["Topic", "Count", "Ratio"])
                        topics_df = topics_df.sort_values(by="Ratio", ascending=False).reset_index(drop=True)
                        if len(topics_df) < 50:
                            missing = 50 - len(topics_df)
                            pad_df = pd.DataFrame([["", 0, 0.0]] * missing, columns=["Topic", "Count", "Ratio"])
                            topics_df = pd.concat([topics_df, pad_df], ignore_index=True)
                        else:
                            topics_df = topics_df.iloc[:50].reset_index(drop=True)
                        top_half = topics_df.iloc[:25].reset_index(drop=True)
                        bottom_half = topics_df.iloc[25:50].reset_index(drop=True)
                        combined_topics_df = pd.DataFrame({
                            "Rank": list(range(1, 26)),
                            "Topic": top_half["Topic"],
                            "Count": top_half["Count"],
                            "Ratio": top_half["Ratio"],
                            "Rank (2)": list(range(26, 51)),
                            "Topic (2)": bottom_half["Topic"],
                            "Count (2)": bottom_half["Count"],
                            "Ratio (2)": bottom_half["Ratio"]
                        })
                        custom_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("white_yellow", ["#FFFFFF", "#FFFF99"])
                        max_ratio = max(combined_topics_df["Ratio"].max(), combined_topics_df["Ratio (2)"].max())
                        styled_topics_df = combined_topics_df.style.format({
                            "Ratio": "{:.2f} %",
                            "Ratio (2)": "{:.2f} %"
                        }).background_gradient(subset=["Ratio", "Ratio (2)"],
                                                cmap=custom_cmap,
                                                vmin=0,
                                                vmax=max_ratio)
                        styled_topics_df = styled_topics_df.set_table_styles(
                            [{'selector': 'th.col_heading', 'props': [('font-size', '10pt')]}],
                            overwrite=False
                        )
                        styled_topics_df = styled_topics_df.set_properties(**{'font-size': '10pt'})
                        styled_topics_df = styled_topics_df.set_properties(
                            subset=["Rank", "Rank (2)"],
                            **{'background-color': 'black', 'color': 'white'}
                        ).hide(axis="index")
                        st.markdown(styled_topics_df.to_html(), unsafe_allow_html=True)
                    else:
                        st.info("No topics data available.")
                else:
                    st.error("No enriched record found for the selected institution.")
