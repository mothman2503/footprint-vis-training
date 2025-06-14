import streamlit as st
import pandas as pd
import os

# CONFIG
MISSING_DIR = "balanced_split_output/missing_classes"
OUTPUT_DIR = "output_chunks/output_chunks_manual"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# List available CSVs
csv_files = [f for f in os.listdir(MISSING_DIR) if f.endswith(".csv")]

st.title("📦 Manual Data Review Tool")
selected_csv = st.selectbox("Select a missing class file to review:", csv_files)

if selected_csv:
    file_path = os.path.join(MISSING_DIR, selected_csv)
    df = pd.read_csv(file_path)

    # Ensure required columns
    df["source"] = df.get("source", "zero_shot")
    df["confidence"] = df.get("confidence", 0.0)

    # Add selection column
    df["✅ Approve"] = False

    st.write(f"### Review and select rows for: {selected_csv}")
    edited_df = st.data_editor(
        df,
        use_container_width=True,
        num_rows="dynamic",
        key="editor"
    )

    # Filter approved rows
    approved_rows = edited_df[edited_df["✅ Approve"] == True].copy()

    if st.button("💾 Save Selected Rows"):
        if not approved_rows.empty:
            approved_rows["source"] = "manual"
            approved_rows["confidence"] = 0.9
            approved_rows.drop(columns=["✅ Approve"], inplace=True)

            output_path = os.path.join(OUTPUT_DIR, selected_csv)
            approved_rows.to_csv(output_path, index=False)
            st.success(f"✅ Saved {len(approved_rows)} approved rows to {output_path}")
        else:
            st.warning("⚠️ No rows selected. Check the boxes in the '✅ Approve' column.")
