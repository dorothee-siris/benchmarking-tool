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
                
                # Rest of your analysis code here, converted to use st.write instead of display
                df_inst = df_master[(df_master["institution"] == institution_name) &
                                  (df_master["country"] == country_code)]
                
                if df_inst.empty:
                    st.error(f"No ranking data found for {institution_name} ({country_code}).")
                    return

                # Your existing analysis code here, adapted for Streamlit
                totals = df_master.groupby(["name of the ranking", "year"]).size().reset_index(name="total")
                total_dict = {(row["name of the ranking"], row["year"]): row["total"]
                             for _, row in totals.iterrows()}
                
                # ... [Continue with your existing analysis code, replacing display() with st.write()]
                # For matplotlib figures:
                # st.pyplot(fig)
                
                # For DataFrames:
                # st.dataframe(df, use_container_width=True)
                
                # For text:
                # st.write("Your text here")

if __name__ == "__main__":
    main()