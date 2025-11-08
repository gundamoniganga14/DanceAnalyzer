# src/main.py
import streamlit as st
import cv2
import os
import analyzer
import pandas as pd

st.set_page_config(page_title="Performance Analyzer ", layout="wide")

# --- Title and Description ---
st.title("💃 Performance Analyzer")
st.markdown("""
Welcome to Performance Analyzer! This application analyzes dance videos with multiple performers and provides a data-driven performance leaderboard.
Simply upload a video, and the application will:
1.  Analyze the movements and grace of each dancer.
2.  Assign a score to each performer.
3.  Generate a leaderboard showing their ranks.
4.  **Provide a snapshot of the single top-performing dancer's best moment.**
""")

# --- Video Uploader and Analysis Button ---
uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    
    if not os.path.exists("data/input_videos"):
        os.makedirs("data/input_videos")

    # Save the uploaded file temporarily
    video_path = os.path.join("data/input_videos", uploaded_file.name)
    with open(video_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.video(video_path)

    if st.button("Start Analysis"):
        with st.spinner('Analyzing dance performance... This may take a few minutes.'):
            # Perform the analysis using the analyzer.py module
            try:
                analysis_results = analyzer.analyze_performance(video_path)
                
                st.balloons()
                st.success("Analysis complete!")
                
                # --- Display Leaderboard ---
                st.header("🏆 Performance Leaderboard")
                
                leaderboard_df = pd.DataFrame(analysis_results['leaderboard'])
                st.dataframe(leaderboard_df.set_index('Rank'))
            
                st.header("📸 Top Performer Snapshot")
                
                snapshot_info = analysis_results.get('best_performer_snapshot', {})
                if snapshot_info and snapshot_info.get('path'):
                    st.image(snapshot_info['path'], 
                             caption=f"Best moment captured from **{snapshot_info['dancer_id']}**")
                else:
                    st.warning("No best moments were captured.")

                st.markdown("---")
                st.info("Thank you for using the Performance Analyzer! We hope this helps in evaluating dance performances.")

            except Exception as e:
                st.error(f"An error occurred during analysis: {e}")