import io, re, base64, datetime
from pathlib import Path
import pandas as pd
import numpy as np
from dash import Dash, html, dcc, Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
from PIL import Image


CSV_PATH = r"curated_cleaned_filled.csv"
PORT = 8050
df = pd.read_csv(CSV_PATH)
df = df.copy()
#data preprocessing
# timestamps
if "timestamp" in df.columns:
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

# numeric columns
def tonum(col):
    if col and col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").fillna(0)
    else:
        return pd.Series(0, index=df.index)

df["likesCount"] = tonum("likesCount")
df["commentsCount"] = tonum("commentsCount")
df["Engagement"] = df["likesCount"] + df["commentsCount"]

# ensure essential text columns exist
df["caption"] = df.get("caption", "").fillna("").astype(str)
df["Field"] = df.get("Field", "Unknown").fillna("Unknown").astype(str)
df["Influencer_Name"] = df.get("Influencer_Name", "Unknown").fillna("Unknown").astype(str)
df["type"] = df.get("type", "Unknown").fillna("Unknown").astype(str)

# weekday/month features for timing chapter
if "timestamp" in df.columns:
    df["Hour"] = df["timestamp"].dt.hour
    df["DayOfWeek"] = df["timestamp"].dt.day_name()
    df["Month"] = df["timestamp"].dt.month_name()
else:
    df["Hour"] = np.nan
    df["DayOfWeek"] = "Unknown"
    df["Month"] = "Unknown"

# interaction metrics
df["CommentToLikeRatio"] = np.where(df["likesCount"] > 0, df["commentsCount"] / df["likesCount"], 0)
df["DiscussionIntensity"] = df["commentsCount"] / df["Engagement"].replace(0, np.nan)
df["ViralPotential"] = (df["Engagement"] ** 0.5) * (1 + df["CommentToLikeRatio"])

# utility: extract hashtags & words
def extract_hashtags(text):
    if not isinstance(text, str): return []
    return re.findall(r"#\w+", text.lower())

def extract_words(text, strip_hashtags=True):
    if not isinstance(text, str): return []
    toks = re.findall(r"[A-Za-z0-9#@']+", text.lower())
    stop = set("a an the and or but if in on for with to of is are this that it at by from as was were be been has have will would can should i you he she we they".split())
    out = []
    for t in toks:
        if t.startswith("@"): continue
        if strip_hashtags and t.startswith("#"): continue
        if len(t) <= 2: continue
        if t in stop: continue
        out.append(t)
    return out

rows = []
for _, r in df.iterrows():
    tags = extract_hashtags(r["caption"])
    for t in tags:
        rows.append({
            "Hashtag": t,
            "Engagement": r["Engagement"],
            "likesCount": r["likesCount"],
            "commentsCount": r["commentsCount"],
            "Influencer_Name": r["Influencer_Name"],
            "Field": r["Field"]
        })
hash_df = pd.DataFrame(rows)

#creating app usign dash and html
app = Dash(__name__, suppress_callback_exceptions=True)
server = app.server

influencer_options = ["All"] + sorted(df["Influencer_Name"].dropna().unique().tolist())
field_options = ["All"] + sorted(df["Field"].dropna().unique().tolist())
type_options = ["All"] + sorted(df["type"].dropna().unique().tolist())

# layout
app.layout = html.Div(style={"fontFamily":"Inter, Arial, sans-serif", "margin":"14px"}, children=[
    html.H1("Instagram — Full Interactive Storyboard", style={"textAlign":"center"}),
    html.Div([
        html.Div([
            html.Label("Influencer"),
            dcc.Dropdown(id="filter-influencer", options=[{"label":i,"value":i} for i in influencer_options], value="All")
        ], style={"width":"24%", "display":"inline-block", "paddingRight":"10px"}),

        html.Div([
            html.Label("Field"),
            dcc.Dropdown(id="filter-field", options=[{"label":f,"value":f} for f in field_options], value="All")
        ], style={"width":"24%", "display":"inline-block", "paddingRight":"10px"}),

        html.Div([
            html.Label("Post Type"),
            dcc.Dropdown(id="filter-type", options=[{"label":t,"value":t} for t in type_options], value="All")
        ], style={"width":"24%", "display":"inline-block", "paddingRight":"10px"}),

        html.Div([
            html.Label("Top N"),
            dcc.Input(id="filter-topn", type="number", value=12, min=3, max=100, step=1)
        ], style={"width":"24%", "display":"inline-block"})
    ], style={"marginBottom":"18px"}),

    dcc.Tabs(id="tabs", value="tab-1", children=[
        dcc.Tab(label="Chapter 1 — Audience Influence", value="tab-1"),
        dcc.Tab(label="Chapter 2 — Hashtag Power", value="tab-2"),
        dcc.Tab(label="Chapter 3 — Sentiment & Emotion", value="tab-3"),
        dcc.Tab(label="Chapter 4 — Timing Insights", value="tab-4"),
        dcc.Tab(label="Chapter 5 — Content Format", value="tab-5"),
        dcc.Tab(label="Chapter 6 — Field Comparison", value="tab-6"),
        dcc.Tab(label="Chapter 7 — Interaction Dynamics", value="tab-7"),
    ]),
    html.Div(id="tab-content", style={"marginTop":"16px"})
])

# ---------- helper to filter df ----------
def apply_filters(df_in, influencer, field, ptype):
    dff = df_in.copy()
    if influencer and influencer != "All":
        dff = dff[dff["Influencer_Name"] == influencer]
    if field and field != "All":
        dff = dff[dff["Field"] == field]
    if ptype and ptype != "All":
        dff = dff[dff["type"] == ptype]
    return dff

# create wordcloud image data URI from counts
def make_wordcloud_datauri(counts_dict, width=800, height=360, bg="white"):
    if not counts_dict:
        img = Image.new('RGB', (width, height), color=(240,240,240))
        b = io.BytesIO(); img.save(b, format='PNG'); b.seek(0)
        return 'data:image/png;base64,' + base64.b64encode(b.getvalue()).decode()
    wc = WordCloud(width=width, height=height, background_color=bg, collocations=False)
    wc.generate_from_frequencies(counts_dict)
    b = io.BytesIO(); wc.to_image().save(b, format='PNG'); b.seek(0)
    return 'data:image/png;base64,' + base64.b64encode(b.getvalue()).decode()

# Single callback to populate tab content using current filters
@app.callback(
    Output("tab-content", "children"),
    Input("tabs", "value"),
    Input("filter-influencer", "value"),
    Input("filter-field", "value"),
    Input("filter-type", "value"),
    Input("filter-topn", "value"),
)
def render_tab(tab, influencer, field, ptype, topn):
    topn = int(topn) if topn else 12
    dff = apply_filters(df, influencer, field, ptype)
    # shared safe columns
    dff = dff.copy()
    dff["_likes_safe"] = dff["likesCount"].clip(lower=0).fillna(0)
    dff["_comments_safe"] = dff["commentsCount"].clip(lower=0).fillna(0)
    dff["_eng"] = dff["Engagement"].clip(lower=0).fillna(0)

    if tab == "tab-1":
        # Chapter 1: Audience Influence
        # Top influencers by avg engagement
        infl = dff.groupby("Influencer_Name", as_index=False)["_eng"].mean().sort_values("_eng", ascending=False).head(topn)
        fig1 = px.bar(infl, x="_eng", y="Influencer_Name", orientation="h", title=f"Top {topn} Influencers by Average Engagement", color="_eng", color_continuous_scale="Tealgrn")
        fig1.update_layout(height=520)

        # Engagement distribution by Field (box)
        fig2 = px.box(dff, x="Field", y="_eng", title="Engagement Distribution by Field")
        fig2.update_layout(height=520)

        # Likes vs Comments scatter (bubble)
        fig3 = px.scatter(dff, x="_likes_safe", y="_comments_safe", color="Field", size="_eng", hover_data=["Influencer_Name","type"], title="Likes vs Comments (bubble = engagement)", size_max=20)
        fig3.update_layout(height=520)

        return html.Div([
            dcc.Graph(figure=fig1),
            html.Div(style={"display":"flex","gap":"10px"}, children=[
                html.Div(dcc.Graph(figure=fig2), style={"flex":"1"}),
                html.Div(dcc.Graph(figure=fig3), style={"flex":"1"})
            ])
        ])

    if tab == "tab-2":
        # Chapter 2: Hashtag Power
        if hash_df.empty:
            return html.Div("No hashtags found in dataset.")
        # filter hash_df similarly
        hdf = hash_df.copy()
        if influencer != "All":
            hdf = hdf[hdf["Influencer_Name"] == influencer]
        if field != "All":
            hdf = hdf[hdf["Field"] == field]
        if hdf.empty:
            return html.Div("No hashtags for selected filters.")

        tag_stats = hdf.groupby("Hashtag", as_index=False).agg({"Engagement":"mean","likesCount":"mean","commentsCount":"mean","Influencer_Name":"count"})
        tag_stats = tag_stats.rename(columns={"Influencer_Name":"UsageCount"}).sort_values("Engagement", ascending=False).head(topn)

        fig1 = px.bar(tag_stats, x="Engagement", y="Hashtag", orientation="h", color="Engagement", title=f"Top {topn} Hashtags by Engagement")
        fig1.update_layout(height=500)

        # treemap for popularity
        pop = tag_stats.copy()
        pop["TotalEngagement"] = pop["Engagement"] + pop["likesCount"] + pop["commentsCount"]
        fig2 = px.treemap(pop, path=["Hashtag"], values="TotalEngagement", color="Engagement", color_continuous_scale="Viridis", title="Hashtag Popularity Treemap")
        fig2.update_layout(height=600)

        # freq vs engagement scatter 
        freq = hdf["Hashtag"].value_counts().rename_axis("Hashtag").reset_index(name="Frequency")
        freq = freq.merge(hdf.groupby("Hashtag", as_index=False)["Engagement"].mean(), on="Hashtag").sort_values("Frequency", ascending=False).head(50)
        fig3 = px.scatter(freq, x="Frequency", y="Engagement", text="Hashtag", size="Frequency", title="Hashtag Frequency vs Engagement", size_max=35)
        fig3.update_traces(textposition="top center")
        fig3.update_layout(height=600)

        return html.Div([
            dcc.Graph(figure=fig1),
            dcc.Graph(figure=fig2),
            dcc.Graph(figure=fig3)
        ])

    if tab == "tab-3":
        # Chapter 3: Sentiment & Emotion (use TextBlob if available, else lightweight fallback)
        try:
            from textblob import TextBlob
            TEXTBLOB_AVAILABLE = True
        except Exception:
            TextBlob = None
            TEXTBLOB_AVAILABLE = False

        def sentiment_score(s):
            if s is None:
                return 0.0
            if TEXTBLOB_AVAILABLE:
                try:
                    return TextBlob(str(s)).sentiment.polarity
                except Exception:
                    return 0.0
            # lightweight fallback: count positive vs negative keywords
            txt = str(s).lower()
            pos_words = {"good","great","love","happy","joy","awesome","best","amazing","win","success","beautiful","nice"}
            neg_words = {"bad","sad","hate","angry","worst","terrible","awful","fail","pain","loss","ugly"}
            tokens = re.findall(r"[a-zA-Z0-9']+", txt)
            if not tokens:
                return 0.0
            pos = sum(1 for t in tokens if t in pos_words)
            neg = sum(1 for t in tokens if t in neg_words)
            score = (pos - neg) / max(len(tokens), 1)
            # clamp to [-1,1]
            return max(-1.0, min(1.0, score))
        # compute on filtered dff to speed up
        dff["SentimentScore"] = dff["caption"].apply(sentiment_score)
        def label_sent(x):
            if x > 0.2: return "Positive"
            if x < -0.2: return "Negative"
            return "Neutral"
        dff["SentimentLabel"] = dff["SentimentScore"].apply(label_sent)

        # Emotion simple rules
        def detect_emotion(text):
            text = str(text).lower()
            keywords = {
                "motivational":["success","dream","goal","achieve","believe","motivat"],
                "happy":["happy","smile","joy","fun","love"],
                "sad":["sad","pain","miss","alone","tears"],
                "angry":["angry","hate","furious"],
                "peaceful":["calm","peace","relax"]
            }
            for e, kws in keywords.items():
                for w in kws:
                    if w in text:
                        return e.capitalize()
            return "Neutral"
        dff["Emotion"] = dff["caption"].apply(detect_emotion)

        # avg engagement by sentiment label
        sent = dff.groupby("SentimentLabel", as_index=False)["Engagement"].mean()
        fig1 = px.bar(sent, x="SentimentLabel", y="Engagement", color="SentimentLabel", title="Avg Engagement by Sentiment")
        fig1.update_layout(height=480)

        # emotion vs engagement
        emo = dff.groupby("Emotion", as_index=False)["Engagement"].mean().sort_values("Engagement", ascending=False)
        fig2 = px.bar(emo, x="Emotion", y="Engagement", color="Emotion", title="Avg Engagement by Emotion")
        fig2.update_layout(height=480)

        # sentiment vs engagement scatter
        fig3 = px.scatter(dff, x="SentimentScore", y="Engagement", color="Emotion", hover_data=["Influencer_Name","caption"], size=dff["likesCount"].clip(lower=0)+1, title="Sentiment Score vs Engagement")
        fig3.update_layout(height=600)

        # wordcloud for caption words (top)
        words = []
        for t in dff["caption"].dropna().astype(str):
            words += extract_words(t)
        wcounts = pd.Series(words).value_counts().head(200).to_dict()
        wc_uri = make_wordcloud_datauri(wcounts, width=900, height=360)

        return html.Div([
            dcc.Graph(figure=fig1),
            dcc.Graph(figure=fig2),
            dcc.Graph(figure=fig3),
            html.Img(src=wc_uri, style={"width":"100%","border":"1px solid #eee","marginTop":"10px"})
        ])

    if tab == "tab-4":
        # Chapter 4: Timing Insights
        if "timestamp" not in dff.columns or dff["timestamp"].isna().all():
            return html.Div("No timestamp info available in dataset for timing analysis.")
        hour_avg = dff.groupby("Hour", as_index=False)["Engagement"].mean().fillna(0)
        fig1 = px.line(hour_avg, x="Hour", y="Engagement", markers=True, title="Avg Engagement by Hour")
        fig1.update_layout(xaxis=dict(dtick=1), height=420)

        day_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        day_avg = dff.groupby("DayOfWeek", as_index=False)["Engagement"].mean()
        day_avg["DayOfWeek"] = pd.Categorical(day_avg["DayOfWeek"], categories=day_order, ordered=True)
        day_avg = day_avg.sort_values("DayOfWeek")
        fig2 = px.bar(day_avg, x="DayOfWeek", y="Engagement", title="Avg Engagement by Day of Week")
        fig2.update_layout(height=420)

        month_order = ["January","February","March","April","May","June","July","August","September","October","November","December"]
        month_avg = dff.groupby("Month", as_index=False)["Engagement"].mean()
        month_avg["Month"] = pd.Categorical(month_avg["Month"], categories=month_order, ordered=True)
        month_avg = month_avg.sort_values("Month")
        fig3 = px.bar(month_avg, x="Month", y="Engagement", title="Avg Engagement by Month")
        fig3.update_layout(height=420)

        heat = dff.groupby(["DayOfWeek","Hour"], as_index=False)["Engagement"].mean()
        heat["DayOfWeek"] = pd.Categorical(heat["DayOfWeek"], categories=day_order, ordered=True)
        heat = heat.sort_values(["DayOfWeek","Hour"])
        fig4 = px.density_heatmap(heat, x="Hour", y="DayOfWeek", z="Engagement", color_continuous_scale="Turbo", title="Engagement Heatmap (Day vs Hour)")
        fig4.update_layout(height=600)

        return html.Div([
            dcc.Graph(figure=fig1),
            dcc.Graph(figure=fig2),
            dcc.Graph(figure=fig3),
            dcc.Graph(figure=fig4)
        ])

    if tab == "tab-5":
        # Chapter 5: Content Format Effectiveness
        type_avg = dff.groupby("type", as_index=False).agg({"Engagement":"mean","likesCount":"mean","commentsCount":"mean"}).sort_values("Engagement", ascending=False)
        fig1 = px.bar(type_avg, x="type", y="Engagement", color="type", title="Avg Engagement by Post Type")
        fig1.update_layout(height=420)

        fig2 = px.violin(dff, x="type", y="Engagement", color="type", box=True, points="all", title="Engagement Distribution by Post Type")
        fig2.update_layout(height=520)

        # likes vs comments bubble by type
        type_metrics = dff.groupby("type", as_index=False).agg({"likesCount":"mean","commentsCount":"mean","Engagement":"mean"})
        fig3 = px.scatter(type_metrics, x="likesCount", y="commentsCount", size="Engagement", color="type", text="type", size_max=40, title="Likes vs Comments per Post Type")
        fig3.update_traces(textposition="top center")
        fig3.update_layout(height=520)

        # type performance across all fields when global
        if influencer == "All" and field == "All":
            ft = df.groupby(["Field","type"], as_index=False)["Engagement"].mean()
            fig4 = px.bar(ft, x="Field", y="Engagement", color="type", barmode="group", title="Post Type Performance by Field")
            fig4.update_layout(height=600)
            return html.Div([dcc.Graph(figure=fig1), dcc.Graph(figure=fig2), dcc.Graph(figure=fig3), dcc.Graph(figure=fig4)])
        else:
            return html.Div([dcc.Graph(figure=fig1), dcc.Graph(figure=fig2), dcc.Graph(figure=fig3)])

    if tab == "tab-6":
        # Chapter 6: Field-based Comparison
        field_avg = df.groupby("Field", as_index=False).agg({"Engagement":"mean","likesCount":"mean","commentsCount":"mean"}).sort_values("Engagement", ascending=False)
        fig1 = px.bar(field_avg, x="Field", y="Engagement", color="Engagement", title="Average Engagement per Field")
        fig1.update_layout(height=520)

        # top influencers in this field (if filtered)
        if field != "All":
            top_infl = dff.groupby("Influencer_Name", as_index=False)["Engagement"].mean().sort_values("Engagement", ascending=False).head(topn)
            fig2 = px.bar(top_infl, x="Engagement", y="Influencer_Name", orientation="h", title=f"Top {topn} Influencers in {field}")
            fig2.update_layout(height=520)
        else:
            fig2 = go.Figure(); fig2.update_layout(title="Select a field to see top influencers", height=250)

        fig3 = px.scatter(field_avg, x="likesCount", y="commentsCount", size="Engagement", text="Field", title="Likes vs Comments per Field")
        fig3.update_traces(textposition="top center")
        fig3.update_layout(height=520)

        sun_df = df.groupby(["Field","type"], as_index=False)["Engagement"].mean()
        fig4 = px.sunburst(sun_df, path=["Field","type"], values="Engagement", color="Engagement", title="Field & Post Type Composition")
        fig4.update_layout(height=600)

        return html.Div([dcc.Graph(figure=fig1), dcc.Graph(figure=fig2), dcc.Graph(figure=fig3), dcc.Graph(figure=fig4)])

    if tab == "tab-7":
        # Chapter 7: Audience Interaction Dynamics
        infl_ratio = dff.groupby("Influencer_Name", as_index=False)["CommentToLikeRatio"].mean().sort_values("CommentToLikeRatio", ascending=False).head(topn)
        fig1 = px.bar(infl_ratio, x="CommentToLikeRatio", y="Influencer_Name", orientation="h", color="CommentToLikeRatio", title=f"Top {topn} Influencers by Comment-to-Like Ratio")
        fig1.update_layout(height=520)

        field_eng = dff.groupby("Field", as_index=False).agg({"CommentToLikeRatio":"mean","Engagement":"mean"})
        fig2 = px.scatter(field_eng, x="Engagement", y="CommentToLikeRatio", size="Engagement", color="Field", text="Field", title="Field Interaction Intensity vs Engagement")
        fig2.update_traces(textposition="top center")
        fig2.update_layout(height=520)

        fig3 = px.violin(dff, x="type", y="DiscussionIntensity", color="type", box=True, points="all", title="Discussion Intensity by Post Type")
        fig3.update_layout(height=520)

        viral = dff.groupby("Influencer_Name", as_index=False)["ViralPotential"].mean().sort_values("ViralPotential", ascending=False).head(topn)
        fig4 = px.bar(viral, x="ViralPotential", y="Influencer_Name", orientation="h", color="ViralPotential", title=f"Top {topn} Influencers by Viral Potential")
        fig4.update_layout(height=520)

        corr_cols = ["likesCount","commentsCount","Engagement","CommentToLikeRatio","DiscussionIntensity","ViralPotential"]
        corr = dff[corr_cols].corr().round(2)
        fig5 = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r", title="Correlation Matrix")
        fig5.update_layout(height=520)

        return html.Div([dcc.Graph(figure=fig1), dcc.Graph(figure=fig2), dcc.Graph(figure=fig3), dcc.Graph(figure=fig4), dcc.Graph(figure=fig5)])

    return html.Div("Tab not implemented")

# run server
if __name__ == "__main__":
    print(f"Running app on http://127.0.0.1:{PORT}")
    app.run(debug=True, port=PORT)