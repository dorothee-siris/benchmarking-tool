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

# Custom CSS for buttons
st.markdown("""
    <style>
        div.stButton > button:first-child {
            background-color: #ef476f;
            color: white !important;  /* Force white color */
            font-weight: bold;
        }
        div.stButton > button:first-child:hover {
            background-color: #d03d61;
            color: white !important;
            font-weight: bold;
        }
        /* Add styles for clicked state */
        div.stButton > button:first-child:active, 
        div.stButton > button:first-child:focus {
            background-color: #d03d61;
            color: white !important;
            font-weight: bold;
            border-color: transparent;
        }
        /* Add styles for visited state */
        div.stButton > button:first-child:visited {
            color: white !important;
        }
    </style>
""", unsafe_allow_html=True)

# Custom CSS for small subheaders
st.markdown("""
    <style>
        .small-subheader {
            font-size: 1.5rem;
            font-weight: 600;
            color: white;
            margin-bottom: 0.5rem;
        }
    </style>
""", unsafe_allow_html=True)

# ---------------------------
# Caching Data Loading
# ---------------------------
@st.cache_data
def load_data():
    df_master = pd.read_parquet("data/Scimago_results_2021_2024.parquet")
    df_enriched = pd.read_parquet("data/unique_institutions_enriched.parquet")
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

#definition of European countries
eur_countries = [
                        'ALB', 'AND', 'ARM', 'AUT', 'AZE', 'BEL', 'BIH', 'BLR', 'BGR', 'CHE',
                        'CYP', 'CZE', 'DEU', 'DNK', 'ESP', 'EST', 'FIN', 'FRA','GBR', 'GEO',
                        'GRC', 'HRV', 'HUN', 'IRL', 'ISL', 'ITA', 'KAZ', 'KOS', 'LIE', 'LTU',
                        'LUX', 'LVA', 'MCO', 'MDA', 'MKD', 'MLT', 'MNE', 'NLD', 'NOR', 'POL',
                        'PRT', 'ROU', 'SMR', 'SRB', 'SVK', 'SVN', 'SWE', 'TUR', 'UKR', 'VAT'
                    ]

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

def color_scale(val):
    # Define thresholds (0%, 2%, 4%, 6%)
    thresholds = [0, 2, 4, 6]
    colors = ["#FFFFFF", "#d9bc2b", "#695806", "#332a00"]
    
    # Handle values above max threshold
    if val >= thresholds[-1]:
        return f"background-color: {colors[-1]}; color: white"
    
    # Find the interval where the value falls
    for i in range(len(thresholds)-1):
        if thresholds[i] <= val < thresholds[i+1]:
            # Calculate ratio within this interval
            ratio = (val - thresholds[i]) / (thresholds[i+1] - thresholds[i])
            
            # Convert hex to RGB for interpolation
            c1 = np.array([int(colors[i][j:j+2], 16) for j in (1, 3, 5)])
            c2 = np.array([int(colors[i+1][j:j+2], 16) for j in (1, 3, 5)])
            
            # Interpolate
            c = c1 * (1-ratio) + c2 * ratio
            
            # Convert back to hex
            hex_color = '#%02x%02x%02x' % tuple(c.astype(int))
            
            # Use white text for values >= 4%
            text_color = "white" if val >= 4 else "black"
            return f"background-color: {hex_color}; color: {text_color}"
    
    return f"background-color: {colors[0]}; color: black"

def color_topics_count(val):
    # Define your thresholds and colors
    thresholds = [0, 10, 20, 30]
    colors = ["#FFFFFF", "#d9bc2b", "#695806", "#332a00"]
    
    # Handle values above max threshold
    if val >= thresholds[-1]:
        return f"background-color: {colors[-1]}; color: white"
    
    # Find the interval where the value falls
    for i in range(len(thresholds)-1):
        if thresholds[i] <= val < thresholds[i+1]:
            # Calculate ratio within this interval
            ratio = (val - thresholds[i]) / (thresholds[i+1] - thresholds[i])
            
            # Convert hex to RGB for interpolation
            c1 = np.array([int(colors[i][j:j+2], 16) for j in (1, 3, 5)])
            c2 = np.array([int(colors[i+1][j:j+2], 16) for j in (1, 3, 5)])
            
            # Interpolate
            c = c1 * (1-ratio) + c2 * ratio
            
            # Convert back to hex
            hex_color = '#%02x%02x%02x' % tuple(c.astype(int))
            
            # Use white text for values >= 20
            text_color = "white" if val >= 20 else "black"
            return f"background-color: {hex_color}; color: {text_color}"
    
    return f"background-color: {colors[0]}; color: black"

# Allowing 2-method validation
def find_institution():
    # This function will be called when Enter is pressed or button is clicked
    search_term = st.session_state.institution_search_input.strip().lower()
    if not search_term:
        st.warning("Please enter a non-empty search string.")
        return
    
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

def handle_institution_selection():
    # Check if matches exist in session state
    if "matches" not in st.session_state or not st.session_state.matches:
        st.error("No institutions found.")
        return
    
    # Find the selected institution tuple
    if "matches_dropdown" in st.session_state:
        selected_label = st.session_state.matches_dropdown
        selected_tuple = next((tup for label, tup in st.session_state.matches if label == selected_label), None)
        
        if selected_tuple:
            # Clear any previous benchmark results
            if 'benchmark_df' in st.session_state:
                del st.session_state.benchmark_df
            if 'benchmark_df_raw' in st.session_state:
                del st.session_state.benchmark_df_raw
            
            # Set the current institution
            st.session_state.current_institution = selected_tuple
        else:
            st.error("Unable to find the selected institution.")

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
                styles.append("background-color: #0E1117; color: white; font-size: 14px;")
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
    
    # Remove duplicate columns (if any) before reindexing
    final_df = final_df.loc[:, ~final_df.columns.duplicated()]
    
    # In run_benchmark, reorder before renaming
    expected_cols = [
        'ROR_name', 
        'ROR_country',
        'Scimago_country_code',  # Keep this for filtering
        'appearances', 
        'Total_Publications',
        'similar_topics_count',
        'similar_topics_details',
        'similar_subfields',
        'similar_fields',
        'similar_sdgs',
        'ranking_detail'
    ]
    final_df = final_df.reindex(columns=expected_cols)
    
    # Then do the renaming
    final_df = final_df.rename(columns={
        'ROR_name': 'Institution',
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
        
    # Sort by shared topics count in descending order
    final_df = final_df.sort_values(by='Shared top topics (count)', ascending=False)
    
    # Reset index starting at 1
    final_df.index = range(1, len(final_df) + 1)
    
    # Apply color styling to the topics count column and return the styled dataframe
    styled_df = final_df.style.applymap(
        color_topics_count,
        subset=['Shared top topics (count)']
    )

    # Return both dataframes
    return {'styled': styled_df, 'raw': final_df}

# ---------------------------
# UI: First Section – Display Institution Results
# ---------------------------
st.title("Bench:red[Up]")
st.subheader("On your mark... bench!")

# Multi-line disclaimer with warning icon
st.markdown("""
    ⚠️ **Disclaimer:**  
    The data in BenchUp are partial and should be used as a starting point, not a definitive analysis.  
    While institutions may share similarities based on these indicators, meaningful comparisons require contextual and qualitative assessment.  
    BenchUp helps identify potential international benchmarks but does not provide absolute answers.
""")

col1, col2 = st.columns([1, 3])
with col1:
    # Add a key to the text input to track its value
    search_str = st.text_input("", 
                                placeholder="Enter partial institution name", 
                                key="institution_search_input", 
                                # Add on_change to handle Enter key press
                                on_change=find_institution)

with col2:
    st.markdown('<div style="margin-top: 27px;"></div>', unsafe_allow_html=True)  # Add spacing
    find_button = st.button("Find Your Institution", on_click=find_institution)

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
    # Add key to the selectbox to track its value and enable enter key
    selected_label = st.selectbox(
        "Select Institution", 
        [m[0] for m in st.session_state.matches], 
        key="matches_dropdown",
        # Use on_change to handle selection and clear previous benchmark results
        on_change=handle_institution_selection
    )

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
            st.markdown('<hr style="border: 1px solid #ef476f;">', unsafe_allow_html=True)
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
            st.markdown('<hr style="border: 1px solid #ef476f;">', unsafe_allow_html=True)
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

            st.markdown(f"<b>Total publications for 2015-2024: <span style='color:red'>{total_pubs_str}</span></b>", unsafe_allow_html=True)
            
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
                
                # Fields visualization
                if fields_data:
                    st.markdown('<p class="small-subheader">Top Fields (>5%)</p>', unsafe_allow_html=True)
                    plot_col = st.columns([2, 1])[0]
                    with plot_col:
                        plt.style.use('dark_background')
                        fig_fields, ax_fields = plt.subplots(figsize=(9, 5))
                        fig_fields.set_dpi(100)
                        
                        # Set figure background
                        fig_fields.patch.set_facecolor('#0E1117')
                        ax_fields.set_facecolor('#0E1117')
                        
                        names_fields = [x[0] for x in fields_data]
                        percentages_fields = [x[2] for x in fields_data]
                        bars = ax_fields.barh(names_fields, percentages_fields, color='#16a4d8')
                        
                        ax_fields.set_xlabel("Percentage of 2015-2024 publications", fontsize=10, color='white')
                        ax_fields.xaxis.set_major_formatter(formatter)
                        ax_fields.invert_yaxis()
                        
                        ax_fields.tick_params(axis='both', colors='white')
                        for label in ax_fields.get_xticklabels():
                            label.set_color('white')
                        for label in ax_fields.get_yticklabels():
                            label.set_color('white')
                            
                        for spine in ax_fields.spines.values():
                            spine.set_color('white')
                        
                        for bar, (_, count, _) in zip(bars, fields_data):
                            ax_fields.annotate(
                                f"{count:,}",
                                xy=(bar.get_width(), bar.get_y() + bar.get_height()/2),
                                xytext=(3, 0),
                                textcoords="offset points",
                                va='center',
                                fontsize=9,
                                color='white'
                            )
                            
                        ax_fields.margins(x=0.15)
                        plt.tight_layout()
                        st.pyplot(fig_fields, use_container_width=True)
                else:
                    st.info("No fields data >5%.")

                # Subfields visualization
                if subfields_data:
                    st.markdown('<p class="small-subheader">Top Subfields (>3%)</p>', unsafe_allow_html=True)
                    plot_col = st.columns([2, 1])[0]
                    with plot_col:
                        plt.style.use('dark_background')
                        fig_subfields, ax_subfields = plt.subplots(figsize=(9, 6))
                        fig_subfields.set_dpi(100)
                        
                        # Set figure background
                        fig_subfields.patch.set_facecolor('#0E1117')
                        ax_subfields.set_facecolor('#0E1117')
                        
                        names_subfields = [x[0] for x in subfields_data]
                        percentages_subfields = [x[2] for x in subfields_data]
                        bars = ax_subfields.barh(names_subfields, percentages_subfields, color='#60dbe8')
                        
                        ax_subfields.set_xlabel("Percentage of 2015-2024 publications", fontsize=10, color='white')
                        ax_subfields.xaxis.set_major_formatter(formatter)
                        ax_subfields.invert_yaxis()
                        
                        ax_subfields.tick_params(axis='both', colors='white')
                        for label in ax_subfields.get_xticklabels():
                            label.set_color('white')
                        for label in ax_subfields.get_yticklabels():
                            label.set_color('white')
                            
                        for spine in ax_subfields.spines.values():
                            spine.set_color('white')
                        
                        for bar, (_, count, _) in zip(bars, subfields_data):
                            ax_subfields.annotate(
                                f"{count:,}",
                                xy=(bar.get_width(), bar.get_y() + bar.get_height()/2),
                                xytext=(3, 0),
                                textcoords="offset points",
                                va='center',
                                fontsize=9,
                                color='white'
                            )
                            
                        ax_subfields.margins(x=0.15)
                        plt.tight_layout()
                        st.pyplot(fig_subfields, use_container_width=True)
                else:
                    st.info("No subfields data >3%.")

                # SDGs visualization
                if sdg_data_labeled:
                    st.markdown('<p class="small-subheader">Top SDGs (>1%)</p>', unsafe_allow_html=True)
                    plot_col = st.columns([2, 1])[0]
                    with plot_col:
                        plt.style.use('dark_background')
                        fig_sdgs, ax_sdgs = plt.subplots(figsize=(9, 5))
                        fig_sdgs.set_dpi(100)
                        
                        # Set figure background
                        fig_sdgs.patch.set_facecolor('#0E1117')
                        ax_sdgs.set_facecolor('#0E1117')
                        
                        names_sdgs = [x[0] for x in sdg_data_labeled]
                        percentages_sdgs = [x[2] for x in sdg_data_labeled]
                        bars = ax_sdgs.barh(names_sdgs, percentages_sdgs, color='#9b5fe0')
                        
                        ax_sdgs.set_xlabel("Percentage of 2015-2024 publications", fontsize=10, color='white')
                        ax_sdgs.xaxis.set_major_formatter(formatter)
                        ax_sdgs.invert_yaxis()
                        
                        ax_sdgs.tick_params(axis='both', colors='white')
                        for label in ax_sdgs.get_xticklabels():
                            label.set_color('white')
                        for label in ax_sdgs.get_yticklabels():
                            label.set_color('white')
                            
                        for spine in ax_sdgs.spines.values():
                            spine.set_color('white')
                        
                        for bar, (_, count, _) in zip(bars, sdg_data_labeled):
                            ax_sdgs.annotate(
                                f"{count:,}",
                                xy=(bar.get_width(), bar.get_y() + bar.get_height()/2),
                                xytext=(3, 0),
                                textcoords="offset points",
                                va='center',
                                fontsize=9,
                                color='white'
                            )
                            
                        ax_sdgs.margins(x=0.15)
                        plt.tight_layout()
                        st.pyplot(fig_sdgs, use_container_width=True)
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

                    # Create a column container that takes up about 2/3 of the page width
                    col1, col2 = st.columns([1, 1])  # Creates a 1:1 ratio

                    with col1:
                        st.markdown('<p class="small-subheader">Top 50 Topics</p>', unsafe_allow_html=True)
                        
                        # Apply styling with the updated color_scale function that includes black text
                        styled_df = topics_df.style.applymap(color_scale, subset=['Ratio'])
                        
                        # Display dataframe - now it will fill the column width instead of full page width
                        st.dataframe(
                            styled_df,
                            column_config={
                                "Rank": st.column_config.NumberColumn(
                                    "Rank",
                                    help="Position in the top 50",
                                    format="%d"
                                ),
                                "Topic": st.column_config.TextColumn(
                                    "Topic",
                                    help="Research topic name"
                                ),
                                "Count": st.column_config.NumberColumn(
                                    "Count",
                                    help="Number of publications",
                                    format="%d"
                                ),
                                "Ratio": st.column_config.NumberColumn(
                                    "Ratio",
                                    help="Percentage of total publications",
                                    format="%.2f%%"
                                )
                            },
                            hide_index=True,
                            use_container_width=True  # This will now fill the column instead of the full page
                        )

                        # Put the download button in the same column
                        csv = topics_df.to_csv(index=False)

                        # Clean institution name for filename (remove special characters)
                        clean_institution_name = "".join(c for c in institution_name if c.isalnum() or c in (' ', '-', '_')).strip()

                        st.download_button(
                            label="Download topics as CSV",
                            data=csv,
                            file_name=f"top_50_topics_{clean_institution_name}.csv",
                            mime="text/csv",
                        )
                        st.markdown('<hr style="border: 1px solid #ef476f;">', unsafe_allow_html=True)
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
            
            # check boxes one after the other
            st.checkbox("Europe only", value=True, key="europe_only")
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
            result = run_benchmark(target_key, rank_range, min_appearances)
            
            if result is not None:
                bench_df = result['raw']  # Get the raw dataframe
                styled_df = result['styled']  # Get the styled dataframe
                
                # Apply all filters to the raw dataframe
                bench_df = bench_df[
                    (bench_df['Total publications'] >= st.session_state.min_pubs) &
                    (bench_df['Total publications'] <= st.session_state.max_pubs)
                ].reset_index(drop=True)
                
                # Apply country filters while we still have the country code
                if st.session_state.europe_only:
                    bench_df = bench_df[bench_df['Scimago_country_code'].isin(eur_countries)].reset_index(drop=True)
                
                if st.session_state.exclude_target_country:
                    target_country_code = st.session_state.current_institution[1]
                    bench_df = bench_df[bench_df['Scimago_country_code'] != target_country_code].reset_index(drop=True)

                # Now we can drop the country code column as it's no longer needed
                bench_df = bench_df.drop(columns=['Scimago_country_code'])
                
                bench_df = bench_df.reset_index(drop=True)
                bench_df.index = range(1, len(bench_df)+1)
                
                # Reorder columns
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
                
                # Create new styled version of the filtered dataframe
                styled_df = bench_df.style.applymap(
                    color_topics_count,
                    subset=['Shared top topics (count)']
                )
                
                # Store both versions in session state
                st.session_state.benchmark_df = styled_df  # For display
                st.session_state.benchmark_df_raw = bench_df  # For CSV export
            else:
                st.session_state.benchmark_df = None
                st.session_state.benchmark_df_raw = None

        st.button("Run Benchmark", on_click=run_benchmark_callback)

        # ---------------------------
        # Display Benchmark Results if available
        # ---------------------------
        if "benchmark_df" in st.session_state and st.session_state.benchmark_df is not None:
            st.markdown("<h3>Benchmarking Results</h3>", unsafe_allow_html=True)

            # Add count summary
            result_count = len(st.session_state.benchmark_df_raw)
            st.markdown(f'<span><span style="color: #ef476f">{result_count}</span> similar institution{"s" if result_count != 1 else ""} found.</span>', unsafe_allow_html=True)

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
                        "Total pubs", # Short name for display
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

            if "benchmark_df_raw" in st.session_state and st.session_state.benchmark_df_raw is not None:
                # Prepare CSV download with proper encoding
                inst_name = st.session_state.current_institution[0]
                inst_country_code = st.session_state.current_institution[1]
                
                # Get the country name from df_enriched using the country code
                inst_country = df_enriched[
                    (df_enriched['Institution'] == inst_name) & 
                    (df_enriched['Scimago_country_code'] == inst_country_code)
                ]['ROR_country'].iloc[0]
                
                safe_inst_name = re.sub(r'[^A-Za-z0-9_\-]+', '_', inst_name.strip())
                
                # Create a copy of the dataframe for CSV export
                export_df = st.session_state.benchmark_df_raw.copy()
                
                # Get row count (excluding the empty row we'll add)
                result_count = len(export_df)
                
                # Determine geography and country inclusion text
                geography_text = "from Europe" if st.session_state.europe_only else "across the world"
                country_text = f"excluding {inst_country}" if st.session_state.exclude_target_country else f"including {inst_country}"
                
                # Format publication numbers with thousand separators
                min_pubs_formatted = "{:,}".format(st.session_state.min_pubs)
                max_pubs_formatted = "{:,}".format(st.session_state.max_pubs)
                
                # Add separator row with "-"
                export_df.loc[len(export_df) + 1] = [" "] * len(export_df.columns)
                export_df.loc[len(export_df) + 2] = [" "] * len(export_df.columns)
                
                # Add multi-line summary
                export_df.loc[len(export_df) + 3] = ["Results:"] + [""] * (len(export_df.columns) - 1)
                export_df.loc[len(export_df) + 4] = [f"{result_count} institutions {geography_text} ({country_text}),"] + [""] * (len(export_df.columns) - 1)
                export_df.loc[len(export_df) + 5] = [f"with between {min_pubs_formatted} and {max_pubs_formatted} publications from 2015 to 2024,"] + [""] * (len(export_df.columns) - 1)
                export_df.loc[len(export_df) + 6] = [f"sharing at least {st.session_state.min_appearances} Scimago thematic rankings with {inst_name} and ranking within ±{st.session_state.rank_range} in each."] + [""] * (len(export_df.columns) - 1)

                # Convert to CSV with utf-8-sig encoding
                csv_data = export_df.to_csv(index=False, encoding='utf-8-sig')
                
                # Convert to bytes
                csv_bytes = csv_data.encode('utf-8-sig')
                
                st.download_button(
                    label="Download benchmark results as CSV",
                    data=csv_bytes,
                    file_name=f"benchmark_results_{safe_inst_name}.csv",
                    mime="text/csv",
                )