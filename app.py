import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import textwrap
from matplotlib.colors import LinearSegmentedColormap

# Cache the data loading
@st.cache_data
def load_data():
    df_master = pd.read_csv("data/Scimago_results_2021_2024.csv", low_memory=False)
    df_enriched = pd.read_csv("data/unique_institutions_enriched.csv", encoding="utf-8-sig")
    return df_master, df_enriched

# SDG Numbers Dictionary
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

def parse_topics_string(s):
    if pd.isna(s) or s.strip() == "":
        return []
    pattern = r'(.+?)\s*\((\d+)\)(?:;\s*|$)'
    matches = re.findall(pattern, s)
    return [(match[0].strip(), int(match[1])) for match in matches]

def main():
    st.title("Institution Analysis Dashboard")
    
    # Load data
    df_master, df_enriched = load_data()
    
    # Search box
    search_str = st.text_input("Search for an institution:", "")
    
    if search_str:
        search_str = search_str.strip().lower()
        matches = [(f"{row['Institution']} ({row['Scimago_country_code']})",
                   (row['Institution'], row['Scimago_country_code']))
                  for _, row in df_enriched.iterrows() 
                  if search_str in row['Institution'].lower()]
        
        if not matches:
            st.warning(f"No institutions found containing '{search_str}'.")
        else:
            st.success(f"Found {len(matches)} match(es). Please select one from the dropdown.")
            
            # Convert matches to format for selectbox
            options = {match[0]: match[1] for match in matches}
            selected = st.selectbox("Select institution:", list(options.keys()))
            
            if st.button("Display Results"):
                institution_name, country_code = options[selected]
                
                df_inst = df_master[(df_master["institution"] == institution_name) &
                                  (df_master["country"] == country_code)]
                
                if df_inst.empty:
                    st.error(f"No ranking data found for {institution_name} ({country_code}).")
                    return

                # Calculate rankings data
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
                
                # Create the rankings table
                table_data = []
                summary_counts = {year: 0 for year in years}
                
                for ranking in ranking_names:
                    if not any((ranking, year) in inst_dict for year in years):
                        continue
                    row_data = {"Ranking": ranking}
                    for year in years:
                        key = (ranking, year)
                        if key not in total_dict:
                            row_data[str(year)] = "—"
                        else:
                            total = total_dict[key]
                            if key in inst_dict:
                                cell_val = f"{inst_dict[key]} / {total}"
                                summary_counts[year] += 1
                            else:
                                cell_val = f"— / {total}"
                            row_data[str(year)] = cell_val
                    table_data.append(row_data)
                
                if table_data:
                    df_display = pd.DataFrame(table_data)
                    st.dataframe(df_display, use_container_width=True)
                    
                    # Display summary
                    total_appearances = sum(summary_counts.values())
                    summary_text = f"{institution_name} ({country_code}) appears {total_appearances} times in total: "
                    summary_parts = [f"{summary_counts[year]} in {year}" for year in years]
                    st.write(summary_text + ", ".join(summary_parts) + ".")
                    
                    # Display enriched data
                    df_filtered = df_enriched[(df_enriched["Institution"] == institution_name) & 
                                           (df_enriched["Scimago_country_code"] == country_code)]
                    
                    if not df_filtered.empty:
                        record = df_filtered.iloc[0]
                        total_pubs = record.get("Total_Publications", "no match")
                        fields_str = record.get("fields", "")
                        subfields_str = record.get("Top_30_Subfields", "")
                        sdg_str = record.get("SDG", "")
                        
                        st.write(f"\nTotal Publications (2015-2024): {total_pubs}")
                        
                        # Create visualizations
                        if total_pubs != "no match":
                            try:
                                total_pubs_int = int(total_pubs)
                                
                                # Process fields data
                                fields_data = parse_topics_string(fields_str)
                                if fields_data:
                                    fields_data = [(name.strip(), count, count/total_pubs_int*100) 
                                                 for name, count in fields_data if (count/total_pubs_int*100) > 5]
                                    fields_data = sorted(fields_data, key=lambda x: x[2], reverse=True)
                                    
                                    if fields_data:
                                        st.subheader("Top Fields (>5%)")
                                        fig, ax = plt.subplots(figsize=(10, 6))
                                        names = [x[0] for x in fields_data]
                                        percentages = [x[2] for x in fields_data]
                                        bars = ax.barh(names, percentages, color='skyblue')
                                        
                                        # Add value labels
                                        for bar, (_, count, _) in zip(bars, fields_data):
                                            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, 
                                                  f"{count}", va='center')
                                        
                                        plt.xlabel("Percentage (%)")
                                        plt.title("Top Fields Distribution")
                                        st.pyplot(fig)
                                        plt.close()
                                    
                                # Add similar visualizations for subfields and SDGs...
                                        
                            except ValueError:
                                st.error("Error processing publication data")
                    else:
                        st.warning("No enriched record found for the selected institution.")

if __name__ == "__main__":
    main()