# Decoding Digital Influence: A Data-Driven Analysis of Instagram Engagement Dynamics

A comprehensive analytical dashboard that explores patterns behind influencer engagement, audience behavior, and content performance on Instagram.

## Overview

This project investigates how influencers drive audience engagement on Instagram using data-driven insights. By combining quantitative signals such as likes and comments with qualitative signals such as sentiment, emotion, timing, and niche/category, the dashboard reveals actionable patterns that can help creators, marketers, and analysts better understand what drives interaction.

The interactive dashboard visualizes engagement trends across multiple dimensions, including:
- Emotion and sentiment
- Post type
- timing
- hashtags
- virality
- influencer field/category

## Key Objectives

- Analyze the factors influencing engagement, defined as likes + comments.
- Identify field-specific trends among influencers such as Sports, Music, Lifestyle, and more.
- Examine how sentiment and emotion affect audience response.
- Evaluate posting-time patterns to understand optimal visibility.
- Compare content formats such as video, carousel, and image posts.
- Highlight top influencers by average engagement and comment-to-like ratio.

## Features of the Dashboard

### 1. Temporal Heatmaps
Identify engagement peaks by day and hour.

### 2. Emotion & Sentiment Analysis
Visualize how positivity, tone, and emotional cues affect engagement.

### 3. Post Type Comparison
Explore how videos, carousels, and images perform across different niches.

### 4. Hashtag Efficiency Map
Discover the relationship between hashtag usage and engagement.

### 5. Correlation Matrix
Understand interdependencies between likes, comments, engagement, and virality.

### 6. Influencer Ranking
View dynamic rankings of creators by engagement and interaction quality.

## Technology Stack

| Component | Description |
|---|---|
| Language | Python 3.x |
| Framework | Dash (Plotly) |
| Libraries | pandas, numpy, plotly, wordcloud, Pillow, textblob |

## Project Structure

```bash
instagram-engagement-dashboard/
├── app.py
├── curated_cleaned_filled.csv
├── requirements.txt
└── README.md
```

## Dataset Setup

This project uses a CSV dataset named `curated_cleaned_filled.csv`.

The Dataset is provided as a ZIP file, first unzip it and place the extracted CSV file in the project root folder, along with `app.py`.

Example:

```bash
unzip curated_cleaned_filled.zip
```

After extraction, make sure the CSV file is available here:

```bash
instagram-engagement-dashboard/curated_cleaned_filled.csv
```

If the CSV file is missing or placed in another folder, the dashboard will not start correctly.

## 🔗 Live Demo
 https://huggingface.co/spaces/pritam01/influencer_dashboard


## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/instagram-engagement-dashboard.git
cd instagram-engagement-dashboard
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
```

On Windows:

```bash
venv\\Scripts\\activate
```

On macOS/Linux:

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Ensure the dataset is available

If the dataset is zipped, unzip it first and place `curated_cleaned_filled.csv` in the project root directory.

### 5. Run the dashboard

```bash
python app.py
```

### 6. Open the app in your browser

```bash
http://127.0.0.1:8050/
```

## Key Insights

- Positive content tends to receive higher engagement than neutral or negative posts.
- Carousels often outperform videos and images, highlighting the value of swipe-based storytelling.
- Best posting times are often early mornings and late evenings.
- Micro-influencers may show stronger audience loyalty through higher comment ratios.
- Engagement is not the same as reach: likes often reflect visibility, while comments can reflect trust and deeper interaction.

## Notes

- The dashboard is built for exploratory analysis and can be extended with more advanced NLP, topic modeling, or clustering.
- Some analyses, such as sentiment and emotion detection, use lightweight text-based heuristics when external NLP packages are unavailable.

## Course Information

**Course:** DA2402 – Data Analytics  
**Instructor:** Prof. Dr. Arun Ayyar  
**Department:** Artificial Intelligence & Data Science  
**Institution:** Indian Institute of Technology Madras
