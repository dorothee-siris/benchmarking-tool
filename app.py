import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.ticker as mticker
import re
import textwrap

# ---------------------------
# Set page config to use full width
# ---------------------------
st.set_page_config(
    page_title="Benchmarking tool",
    layout="wide"
)

# Add this at the top of your script, after the st.set_page_config
st.markdown("""
    <style>
        div.stButton > button:first-child {
            background-color: #ef476f;
            color: white;
            font-weight: bold;
        }
        div.stButton > button:first-child:hover {
            background-color: #d03d61;  /* Slightly darker shade for hover effect */
            color: white;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)

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
# SDG Official Numbers Dictionary and Normalization
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
sdg_variants = {
    "peace and strong institution": 16,
    "industry and innovations": 9,
}

def normalize_sdg_key(s):
    return " ".join("".join(ch for ch in s.lower() if ch.isalnum() or ch==" ").split())

sdg_numbers_norm = {normalize_sdg_key(k): v for k, v in sdg_numbers.items()}
sdg_variants_norm = {normalize_sdg_key(k): v for k, v in sdg_variants.items()}

# ---------------------------
# Helper Functions for Display & Parsing
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
    c0 = (239/255, 71/255, 111/255)
    c_mid = (255/255, 209/255, 102/255)
    c1 = (6/255, 214/255, 160/255)
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
    s = str(cell)
    if len(s) > width:
        return s[:width]
    return s.ljust(width)

# ---------------------------
# Heatmap Styling Function
# ---------------------------
def color_cells_dynamic(row):
    styles = []
    for col in row.index:
        if col == "Ranking":
            styles.append("")
        else:
            cell = row[col]
            if pd.isna(cell) or str(cell).strip().lower() == "no data":
                styles.append("background-color: white; color: black; font-size: 16px;")
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
                hex_color = get_heatmap_color(1 - ratio)
                styles.append(f"background-color: {hex_color}; color: black;")
    return styles


# ---------------------------
# Main Benchmarking Function
# ---------------------------
def run_benchmark(target_key, rank_range, min_appearances):
    df_master_2024 = df_master[df_master['year'] == 2024].copy()
    df_master_2024['lookup_key'] = list(zip(df_master_2024['institution'], df_master_2024['country']))
    df_enriched['lookup_key'] = list(zip(df_enriched['Institution'], df_enriched['Scimago_country_code']))
    df_enriched['Total_Publications'] = pd.to_numeric(df_enriched['Total_Publications'], errors='coerce').fillna(0)
    
    target_rows = df_master_2024[df_master_2024['lookup_key'] == target_key]
    if target_rows.empty:
        st.error("Target institution not found in 2024 rankings.")
        return None
    candidate_records = []
    for _, row in target_rows.iterrows():
        ranking_code = row['code']
        ranking_name = row['name of the ranking']
        target_rank = row['rank']
        lower_bound = max(target_rank - rank_range, 1)
        upper_bound = target_rank + rank_range
        candidates = df_master_2024[
            (df_master_2024['code'] == ranking_code) &
            (df_master_2024['rank'] >= lower_bound) &
            (df_master_2024['rank'] <= upper_bound)
        ]
        candidates = candidates[candidates['lookup_key'] != target_key]
        for _, cand in candidates.iterrows():
            candidate_records.append({
                'lookup_key': cand['lookup_key'],
                'institution': cand['institution'],
                'ranking_detail': f"{ranking_name} ({int(cand['rank'])})"
            })
    if not candidate_records:
        st.error("No candidate institutions found within the specified rank range.")
        return None
    candidates_df = pd.DataFrame(candidate_records)
    aggregated = candidates_df.groupby('lookup_key').agg({
        'institution': 'first',
        'ranking_detail': lambda x: "; ".join(x),
        'lookup_key': 'count'
    }).rename(columns={'lookup_key': 'appearances'}).reset_index()
    aggregated = aggregated[aggregated['appearances'] >= min_appearances]
    merged = pd.merge(aggregated, df_enriched, on='lookup_key', how='left')
    
    target_enriched_row = df_enriched[df_enriched['lookup_key'] == target_key]
    if target_enriched_row.empty:
        st.error("Target institution enriched data not found.")
        return None
    target_enriched = target_enriched_row.iloc[0]
    target_total_pubs = float(target_enriched['Total_Publications'])
    
    def parse_metrics(metric_str):
        result = {}
        if pd.isna(metric_str) or metric_str.strip() == "":
            return result
        items = metric_str.split(";")
        for item in items:
            item = item.strip()
            if "(" in item and ")" in item:
                try:
                    open_paren = item.rindex("(")
                    close_paren = item.rindex(")")
                    field_name = item[:open_paren].strip()
                    count_str = item[open_paren+1:close_paren].strip()
                    count_val = float(count_str)
                    result[field_name] = count_val
                except:
                    pass
        return result
    
    def compute_percentage(metrics_dict, total):
        perc_dict = {}
        for k, v in metrics_dict.items():
            perc_dict[k] = (v / total) * 100 if total > 0 else 0
        return perc_dict
    
    target_fields = parse_metrics(target_enriched.get('fields', ''))
    target_fields_perc = compute_percentage(target_fields, target_total_pubs)
    
    target_subfields = parse_metrics(target_enriched.get('Top_30_Subfields', ''))
    target_subfields_perc = compute_percentage(target_subfields, target_total_pubs)
    
    target_topics = parse_metrics(target_enriched.get('Top_50_Topics', ''))
    target_topics_perc = compute_percentage(target_topics, target_total_pubs)
    
    target_sdgs = parse_metrics(target_enriched.get('SDG', ''))
    target_sdgs_perc = compute_percentage(target_sdgs, target_total_pubs)
    
    def compute_similar(row):
        try:
            total_pubs = float(row.get('Total_Publications', 0))
        except:
            total_pubs = 0
        fields_dict = parse_metrics(row.get('fields', ''))
        fields_perc = compute_percentage(fields_dict, total_pubs) if total_pubs > 0 else {}
        similar_fields = {f: fields_perc[f] for f in fields_perc 
                          if f in target_fields_perc and fields_perc[f] > 5 and target_fields_perc[f] > 5}
        similar_fields_sorted = sorted(similar_fields.items(), key=lambda x: x[1], reverse=True)
        similar_fields_str = "; ".join([f"{k} ({v:.1f}%)" for k, v in similar_fields_sorted])
        
        subfields_dict = parse_metrics(row.get('Top_30_Subfields', ''))
        subfields_perc = compute_percentage(subfields_dict, total_pubs) if total_pubs > 0 else {}
        similar_subfields = {sf: subfields_perc[sf] for sf in subfields_perc 
                             if sf in target_subfields_perc and subfields_perc[sf] > 3 and target_subfields_perc[sf] > 3}
        similar_subfields_sorted = sorted(similar_subfields.items(), key=lambda x: x[1], reverse=True)
        similar_subfields_str = "; ".join([f"{k} ({v:.1f}%)" for k, v in similar_subfields_sorted])
        
        topics_dict = parse_metrics(row.get('Top_50_Topics', ''))
        topics_perc = compute_percentage(topics_dict, total_pubs) if total_pubs > 0 else {}
        similar_topics = {t: topics_perc[t] for t in topics_perc if t in target_topics_perc}
        similar_topics_sorted = sorted(similar_topics.items(), key=lambda x: x[1], reverse=True)
        similar_topics_str = "; ".join([f"{k} ({int(topics_dict.get(k, 0))}, {v:.1f}%)" for k, v in similar_topics_sorted])
        similar_topics_count = len(similar_topics)
        
        sdgs_dict = parse_metrics(row.get('SDG', ''))
        sdgs_perc = compute_percentage(sdgs_dict, total_pubs) if total_pubs > 0 else {}
        similar_sdgs = {s: sdgs_perc[s] for s in sdgs_perc
                        if s in target_sdgs_perc and sdgs_perc[s] > 3 and target_sdgs_perc[s] > 3}
        similar_sdgs_sorted = sorted(similar_sdgs.items(), key=lambda x: x[1], reverse=True)
        similar_sdgs_str = "; ".join([f"{k} ({int(sdgs_dict.get(k, 0))}, {v:.1f}%)" for k, v in similar_sdgs_sorted])
        
        return pd.Series({
            'similar_fields': similar_fields_str,
            'similar_subfields': similar_subfields_str,
            'similar_topics_count': similar_topics_count,
            'similar_topics_details': similar_topics_str,
            'similar_sdgs': similar_sdgs_str
        })
    
    similar_metrics = merged.apply(compute_similar, axis=1)
    final_df = pd.concat([merged, similar_metrics], axis=1)
    
    final_df = final_df[[ 
        'ROR_name', 
        'Scimago_country_code', 
        'ROR_country',
        'appearances', 
        'ranking_detail', 
        'Total_Publications',
        'similar_fields', 
        'similar_subfields', 
        'similar_topics_count', 
        'similar_topics_details',
        'similar_sdgs'
    ]]
    
    final_df = final_df.rename(columns={
        'ROR_name': 'Institution',
        'Scimago_country_code': 'Country code',
        'ROR_country': 'Country',
        'appearances': 'Shared rankings (count)',
        'ranking_detail': 'Shared rankings (detail)',
        'Total_Publications': 'Total publications',
        'similar_fields': 'Shared top fields',
        'similar_subfields': 'Shared top subfields',
        'similar_topics_count': 'Shared top topics (count)',
        'similar_topics_details': 'Shared top topics',
        'similar_sdgs': 'Shared top SDGs'
    })
    
    final_df['Total publications'] = final_df['Total publications'].fillna(0).round().astype(int)
    
    return final_df

# ---------------------------
# UI: First Section – Display Institution Results
# ---------------------------
st.title("Bench:red[Up]")
st.header("On your mark... bench!")

col1, col2 = st.columns([1, 3])
with col1:
    search_str = st.text_input(placeholder="Enter partial institution name", key="search_str")
with col2:
    find_button = st.button("Find Matches")

if find_button:
    search_term = search_str.strip().lower()
    if not search_term:
        st.warning("Please enter a non-empty search string.")
    else:
        matches = [
            (f"{row['Institution']} ({row['Scimago_country_code']})",
             (row['Institution'], row['Scimago_country_code']))
            for _, row in df_enriched.iterrows()
            if search_term in row['Institution'].lower()
        ]
        if not matches:
            st.info(f"No institutions found containing '{search_term}'.")
        else:
            st.session_state.matches = matches
            st.success(f"Found {len(matches)} match(es). Please select one below.")

if "matches" in st.session_state:
    selected_label = st.selectbox("Select Institution", [m[0] for m in st.session_state.matches], key="matches_dropdown")
    selected_tuple = next((tup for label, tup in st.session_state.matches if label == selected_label), None)
    if st.button("Display Results"):
        if not selected_tuple:
            st.error("No institution selected.")
        else:
            st.session_state.current_institution = selected_tuple

# Always display first results if an institution has been chosen.
if "current_institution" in st.session_state:
    with st.container():
        institution_name, country_code = st.session_state.current_institution
        df_inst = df_master[
            (df_master["institution"] == institution_name) &
            (df_master["country"] == country_code)
        ]
        if df_inst.empty:
            st.error(f"No ranking data found for {institution_name} ({country_code}).")
        else:
            totals = df_master.groupby(["name of the ranking", "year"]).size().reset_index(name="total")
            total_dict = {(row["name of the ranking"], row["year"]): row["total"] for _, row in totals.iterrows()}
            inst_dict = {(row["name of the ranking"], row["year"]): row["rank"] for _, row in df_inst.iterrows()}
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
            result_df = result_df.fillna("no data")
            for col in years:
                result_df[col] = result_df[col].apply(lambda x: fix_width(x, 11))
            
            # In the Scimago results section:
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
            # Store the number of appearances in 2024 rankings:
            st.session_state.target_appearances = summary_counts[2024]

            st.markdown("<h3>OpenAlex results</h3>", unsafe_allow_html=True)
            try:
                total_pubs_int = int(df_enriched[
                    (df_enriched["Institution"] == institution_name) &
                    (df_enriched["Scimago_country_code"] == country_code)
                ].iloc[0].get("Total_Publications", "no match"))
                total_pubs_str = f"{total_pubs_int:,}"
            except Exception:
                total_pubs_str = "no match"

            # Store these values in session_state:
            st.session_state.target_total_publications = total_pubs_str

            st.markdown(f"<b>Total publications (articles only) for 2015-2024: <span style='color:red'>{total_pubs_str}</span></b>", unsafe_allow_html=True)
            
            # Additional enrichment and histograms (fields, subfields, SDGs, topics) as before...
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
                    norm_key = normalize_sdg_key(name)
                    if norm_key in sdg_numbers_norm:
                        number = sdg_numbers_norm[norm_key]
                    elif norm_key in sdg_variants_norm:
                        number = sdg_variants_norm[norm_key]
                    else:
                        number = "?"
                    new_label = f"{name} (SDG {number})"
                    sdg_data_labeled.append((new_label, count, perc))
                
                formatter = mticker.FuncFormatter(lambda x, pos: f"{int(round(x))} %")
                
                if fields_data:
                    st.subheader("Top Fields (>5%)")
                    fig_fields, ax_fields = plt.subplots(figsize=(10, 5))
                    fig_fields.set_dpi(100)
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
                    plt.tight_layout()
                    st.pyplot(fig_fields, use_container_width=False)
                else:
                    st.info("No fields data >5%.")
                
                if subfields_data:
                    st.subheader("Top Subfields (>3%)")
                    fig_subfields, ax_subfields = plt.subplots(figsize=(10, 6))
                    fig_subfields.set_dpi(100)
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
                    plt.tight_layout()
                    st.pyplot(fig_subfields, use_container_width=False)
                else:
                    st.info("No subfields data >3%.")
                
                if sdg_data_labeled:
                    st.subheader("Top SDGs (>1%)")
                    fig_sdgs, ax_sdgs = plt.subplots(figsize=(10, 5))
                    fig_sdgs.set_dpi(100)
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
                    plt.tight_layout()
                    st.pyplot(fig_sdgs, use_container_width=False)
                else:
                    st.info("No SDGs data >1%.")
                
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

# ---------------------------
# UI: Second Section – Benchmarking Parameters
# ---------------------------
st.markdown("<h3>Benchmarking Parameters</h3>", unsafe_allow_html=True)
if "current_institution" in st.session_state and "target_appearances" in st.session_state and "target_total_publications" in st.session_state:
    target_inst = st.session_state.current_institution[0]
    st.markdown(
        f'''
        <div style="font-size: 1rem; margin: 0.5rem 0;">
            As a reminder, <b>{target_inst}</b> appears in 
            <b style="color: #ef476f">{st.session_state.target_appearances}</b> Scimago thematic rankings in 2024 
            and adds up to <b style="color: #ef476f">{st.session_state.target_total_publications}</b> publications (articles only) 
            for the period 2015-2024.
        </div>
        ''',
        unsafe_allow_html=True
    )

    # Create four columns for the numeric inputs
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.number_input(
            "Ranking distance (±)", 
            value=100, 
            min_value=1, 
            max_value=1000, 
            step=1, 
            key="rank_range",
            help="Maximum distance allowed above or below the benchmarked institution's rank in each ranking"
        )
    
    with col2:
        st.number_input(
            "Min. shared rankings",
            value=3,
            min_value=1,
            max_value=100,
            step=1,
            key="min_appearances",
            help="Minimum number of rankings that must be shared with the benchmarked institution"
        )
    
    with col3:
        st.number_input(
            "Min. pubs",
            value=100,
            min_value=0,
            max_value=999999999,
            step=1,
            key="min_pubs",
            help="Minimum amount of publications (articles only) produced over the past 10 years"
        )
    
    with col4:
        st.number_input(
            "Max. pubs",
            value=10000,
            min_value=0,
            max_value=999999999,
            step=1,
            key="max_pubs",
            help="Maximum amount of publications (articles only) produced over the past 10 years"
        )
    
    # Create two columns for the checkboxes
    col_check1, col_check2 = st.columns(2)
    
    with col_check1:
        st.checkbox("Europe only", value=True, key="europe_only")
    
    with col_check2:
        st.checkbox("Exclude target institution country", value=False, key="exclude_target_country")

# ---------------------------
# Callback for Running Benchmark
# ---------------------------
def run_benchmark_callback():
    if "current_institution" not in st.session_state:
        st.error("Please display institution results first.")
        return
    target_key = st.session_state.current_institution
    rank_range = st.session_state.rank_range
    min_appearances = st.session_state.min_appearances
    bench_df = run_benchmark(target_key, rank_range, min_appearances)
    if bench_df is not None:
        bench_df = bench_df[
            (bench_df['Total publications'] >= st.session_state.min_pubs) &
            (bench_df['Total publications'] <= st.session_state.max_pubs)
        ].reset_index(drop=True)
        if st.session_state.europe_only:
            eur_countries = [
                'ALB', 'AND', 'ARM', 'AUT', 'AZE', 'BEL', 'BIH', 'BLR', 'BGR', 'CHE',
                'CYP', 'CZE', 'DEU', 'DNK', 'ESP', 'EST', 'FIN', 'FRA','GBR', 'GEO',
                'GRC', 'HRV', 'HUN', 'IRL', 'ISL', 'ITA', 'KAZ', 'KOS', 'LIE', 'LTU',
                'LUX', 'LVA', 'MCO', 'MDA', 'MKD', 'MLT', 'MNE', 'NLD', 'NOR', 'POL',
                'PRT', 'ROU', 'SMR', 'SRB', 'SVK', 'SVN', 'SWE', 'TUR', 'UKR', 'VAT'
            ]
            bench_df = bench_df[bench_df['Country code'].isin(eur_countries)].reset_index(drop=True)
        if st.session_state.exclude_target_country:
            target_country_code = st.session_state.current_institution[1]  # second element is the country code
            bench_df = bench_df[bench_df['Country code'] != target_country_code].reset_index(drop=True)
        bench_df = bench_df.reset_index(drop=True)
        bench_df.index = range(1, len(bench_df)+1)
        # Drop the "Country code" column
        bench_df = bench_df.drop(columns=['Country code'])
        # Reorder and rename columns as required:
        final_order = [
            'Institution',
            'Country',
            'Shared rankings (count)',
            'Total publications',
            'Shared top topics (count)',
            'Shared top topics',
            'Shared top subfields',
            'Shared top fields',
            'Shared top SDGs',
            'Shared rankings (detail)'
        ]
        bench_df = bench_df[final_order]
        st.session_state.benchmark_df = bench_df
    else:
        st.session_state.benchmark_df = None

st.button("Run Benchmark", on_click=run_benchmark_callback)

# ---------------------------
# Display Benchmark Results if available
# ---------------------------
if "benchmark_df" in st.session_state and st.session_state.benchmark_df is not None:
    st.markdown("<h3>Benchmarking Results</h3>", unsafe_allow_html=True)
    st.dataframe(
        st.session_state.benchmark_df,
        use_container_width=True,
        column_config={
            "Institution": st.column_config.Column(
                width="medium"
            ),
            "Country": st.column_config.Column(
                width="small"
            ),
            "Shared rankings (count)": st.column_config.Column(
                width="small",
                help="Number of Scimago thematic rankings shared with the benchmarked institution in 2024"
            ),
            "Shared rankings (detail)": st.column_config.Column(
                width="small",
                help="List of shared Scimago thematic rankings with rank position in each"
            ),
            "Total publications": st.column_config.Column(
                width="small",
                help="Total number of articles published between 2015-2024, as referenced in OpenAlex"
            ),
            "Shared top fields": st.column_config.Column(
                width="small",
                help="List of research 'fields' (OpenAlex low granularity level) that represent more than 5% of publications for both institutions"
            ),
            "Shared top subfields": st.column_config.Column(
                width="small",
                help="List of research 'subfields'(OpenAlex medium granularity level) that represent more than 3% of publications for both institutions"
            ),
            "Shared top topics (count)": st.column_config.Column(
                width="small",
                help="Number of research topics shared between the top 50 topics of both institutions"
            ),
            "Shared top topics": st.column_config.Column(
                width="medium",
                help="List of shared OpenAlex 'topics' (high granularity level) from the top 50 most frequent topics of both institutions"
            ),
            "Shared top SDGs": st.column_config.Column(
                width="small",
                help="SDG-tagged publications that represent more than 1% of the total publications for both institutions"
            )
        }
    )