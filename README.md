# ad_analytics
ad_analytics analyzes and optimizes TikTok ad performance using advanced computer vision and data analysis. By employing YOLOv8 for object detection and frame clustering to identify key scenes, this project reveals visual elements common in high-performing ads, helping brands adapt to market trends and improve engagement and conversion rates.

## Project Structure

	•	ad_analytics/: Main directory containing the project’s core scripts and subdirectories for analysis and preprocessing.
	•	init.py: Initialization file for the ad_analytics module.
	•	ad_analysis/: Contains scripts and notebooks for analyzing ad performance and visual elements.
	•	preprocessing/: Contains scripts for preprocessing video ads and extracting keyframes.
	•	videos/: Directory where video ads are stored for analysis.

## Key Scripts

	•	feature_extractor.py: Script for extracting features from video ads using computer vision techniques. It processes keyframes and applies the YOLOv8 model to detect and classify objects within the frames.
	•	key_frames_extractor.py: Script for extracting keyframes from video ads. It uses FFMPEG to convert videos into frames and employs clustering techniques to identify the most representative frames based on color histograms.

## Usage

	1.	Preprocessing Videos: Use key_frames_extractor.py to extract keyframes from video ads. This script converts videos into frames and selects the most representative ones.
	2.	Feature Extraction: Use feature_extractor.py to detect and classify objects within the keyframes. The script applies the YOLOv8 model and extracts visual elements with a confidence score above 50%.
	3.	Ad Analysis: Use the scripts in the ad_analysis directory to analyze the extracted features and identify common visual elements in top-performing ads. This analysis helps in understanding the impact of different visual components on ad performance.
