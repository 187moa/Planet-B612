import pandas as pd
import numpy as np
from scipy.stats import pearsonr, ttest_ind
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import dcc, html, Output, Input
import dash_bootstrap_components as dbc
import io
import base64

# Function to analyze popularity bias
def analyze_popularity_bias(df):
    """
    Analyze whether Planet B612 shows bias against mainstream/popular movies
    """
    # Create a clean copy of the dataframe
    df_clean = df.copy()
    
    # Check and convert IMDb_voters column
    if 'IMDb_voters' in df_clean.columns:
        # Already has the correct column name, just clean the data
        pass
    elif 'IMDb voters' in df_clean.columns:
        # Need to rename and clean the column
        df_clean['IMDb_voters'] = df_clean['IMDb voters'].copy()
        df_clean = df_clean.drop('IMDb voters', axis=1)
    else:
        return {"error": "IMDb voters data not found in the dataset"}
    
    # Convert IMDb_voters to numeric, handling string values with commas
    # First convert to string to handle any non-string data types
    df_clean['IMDb_voters'] = df_clean['IMDb_voters'].astype(str)
    # Remove commas and any other non-numeric characters (keeping decimal points)
    # Using raw string to fix the escape sequence warning
    df_clean['IMDb_voters'] = df_clean['IMDb_voters'].str.replace(',', '').str.replace(r'[^0-9\.]', '', regex=True)
    # Convert to float, with empty strings or invalid values becoming NaN
    df_clean['IMDb_voters'] = pd.to_numeric(df_clean['IMDb_voters'], errors='coerce')
    
    # Drop rows with NaN or invalid values
    df_clean = df_clean.dropna(subset=['IMDb_voters', 'rating', 'IMDb', 'rating_difference'])
    df_clean = df_clean[df_clean['IMDb_voters'] > 0]  # Log10 would fail on zero or negative values
    
    # Calculate log_voters for visualization
    df_clean['log_voters'] = np.log10(df_clean['IMDb_voters'])
    
    # Print diagnostic information
    print(f"Original dataset: {len(df)} movies")
    print(f"After cleaning: {len(df_clean)} movies with valid IMDb voter data")
    
    # Continue with the rest of the function as before
    # Calculate correlation between voter count and rating difference
    correlation, p_value = pearsonr(df_clean['log_voters'], df_clean['rating_difference'])
    
    # Create popularity tiers based on IMDb voters
    popularity_bins = [0, 10000, 100000, 500000, float('inf')]
    popularity_labels = ['Niche (<10K)', 'Moderate (10K-100K)', 'Popular (100K-500K)', 'Mainstream (>500K)']
    
    df_clean['popularity_tier'] = pd.cut(df_clean['IMDb_voters'], bins=popularity_bins, labels=popularity_labels)
    
    # Calculate statistics by popularity tier
    tier_stats = df_clean.groupby('popularity_tier').agg({
        'rating': ['mean', 'count'],
        'IMDb': 'mean',
        'rating_difference': ['mean', 'std']
    })
    
    tier_stats.columns = ['b612_mean', 'count', 'imdb_mean', 'diff_mean', 'diff_std']
    tier_stats = tier_stats.reset_index()
    
    # Calculate t-tests between tiers to see if differences are statistically significant
    niche_movies = df_clean[df_clean['popularity_tier'] == 'Niche (<10K)']['rating_difference']
    mainstream_movies = df_clean[df_clean['popularity_tier'] == 'Mainstream (>500K)']['rating_difference']
    
    # Only perform t-test if both groups have data
    if len(niche_movies) > 0 and len(mainstream_movies) > 0:
        niche_vs_mainstream = ttest_ind(niche_movies, mainstream_movies, equal_var=False)
    else:
        niche_vs_mainstream = (float('nan'), float('nan'))
    
    # Find top rated mainstream and niche movies
    mainstream_movies_df = df_clean[df_clean['popularity_tier'] == 'Mainstream (>500K)'].nlargest(10, 'rating')
    niche_movies_df = df_clean[df_clean['popularity_tier'] == 'Niche (<10K)'].nlargest(10, 'rating')
    
    # Calculate average difference for critically acclaimed movies (IMDb > 8.0) by popularity
    acclaimed = df_clean[df_clean['IMDb'] > 8.0]
    acclaimed_by_tier = acclaimed.groupby('popularity_tier')['rating_difference'].mean() if not acclaimed.empty else pd.Series()
    
    # Return all the analysis results
    return {
        "correlation": correlation,
        "p_value": p_value,
        "tier_stats": tier_stats,
        "niche_vs_mainstream_ttest": niche_vs_mainstream,
        "df_clean": df_clean,
        "mainstream_top_movies": mainstream_movies_df,
        "niche_top_movies": niche_movies_df,
        "acclaimed_by_tier": acclaimed_by_tier
    }
# Create popularity bias visualizations
def create_popularity_correlation_figure(analysis_results):
    """Create a scatter plot showing correlation between popularity and rating difference"""
    df_clean = analysis_results["df_clean"]
    correlation = analysis_results["correlation"]
    p_value = analysis_results["p_value"]
    
    fig = go.Figure()
    
    # Create scatter plot
    fig.add_trace(
        go.Scatter(
            x=df_clean['log_voters'],
            y=df_clean['rating_difference'],
            mode='markers',
            marker=dict(
                size=10,
                color=df_clean['IMDb'],
                colorscale='Viridis',
                colorbar=dict(title="IMDb Rating"),
                opacity=0.7
            ),
            text=df_clean['name'],
            hovertemplate="<b>%{text}</b><br>IMDb Voters: %{customdata[0]:,.0f}<br>" +
                          "Planet B612: %{customdata[1]:.1f}<br>IMDb: %{customdata[2]:.1f}<br>" +
                          "Difference: %{y:.1f}<extra></extra>",
            customdata=np.stack((df_clean['IMDb_voters'], df_clean['rating'], df_clean['IMDb']), axis=-1)
        )
    )
    
    # Add trend line
    z = np.polyfit(df_clean['log_voters'], df_clean['rating_difference'], 1)
    p = np.poly1d(z)
    x_range = np.linspace(df_clean['log_voters'].min(), df_clean['log_voters'].max(), 100)
    
    fig.add_trace(
        go.Scatter(
            x=x_range,
            y=p(x_range),
            mode='lines',
            line=dict(color='red', width=2),
            name=f'Trend Line (y={z[0]:.3f}x+{z[1]:.2f})'
        )
    )
    
    # Add zero line
    fig.add_shape(
        type="line",
        x0=df_clean['log_voters'].min(),
        y0=0,
        x1=df_clean['log_voters'].max(),
        y1=0,
        line=dict(color="black", dash="dash", width=1)
    )
    
    # Add annotation with correlation coefficient
    fig.add_annotation(
        x=0.05,
        y=0.95,
        xref="paper",
        yref="paper",
        text=f"Correlation: {correlation:.3f}<br>p-value: {p_value:.4f}",
        showarrow=False,
        font=dict(size=12),
        bgcolor="white",
        bordercolor="black",
        borderwidth=1,
        borderpad=4
    )
    
    # Add annotations for extreme cases
    extremes = pd.concat([
        df_clean.nlargest(3, 'rating_difference'),
        df_clean.nsmallest(3, 'rating_difference'),
        df_clean.nlargest(3, 'log_voters')
    ]).drop_duplicates()
    
    for _, row in extremes.iterrows():
        fig.add_annotation(
            x=row['log_voters'],
            y=row['rating_difference'],
            text=row['name'] if len(row['name']) < 15 else row['name'][:12] + '...',
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=1,
            arrowcolor="black"
        )
    
    # Customize x-axis ticks to show actual voter numbers
    fig.update_xaxes(
        tickvals=[3, 4, 5, 6, 7],
        ticktext=['1K', '10K', '100K', '1M', '10M']
    )
    
    # Update layout
    fig.update_layout(
        title="Relationship Between Movie Popularity and Rating Difference",
        xaxis_title="Number of IMDb Voters (log scale)",
        yaxis_title="Rating Difference (Planet B612 - IMDb)",
        hovermode="closest",
        height=600,
        margin=dict(l=60, r=60, t=50, b=60)
    )
    
    return fig

def create_popularity_tier_figure(analysis_results):
    """Create a bar chart showing rating differences by popularity tier"""
    tier_stats = analysis_results["tier_stats"]
    
    fig = go.Figure()
    
    # Add bars for Planet B612 ratings
    fig.add_trace(
        go.Bar(
            x=tier_stats['popularity_tier'],
            y=tier_stats['b612_mean'],
            name='Planet B612',
            marker_color='blue',
            opacity=0.7,
            hovertemplate="<b>%{x}</b><br>Avg Rating: %{y:.2f}<br>n=%{customdata}<extra></extra>",
            customdata=tier_stats['count']
        )
    )
    
    # Add bars for IMDb ratings
    fig.add_trace(
        go.Bar(
            x=tier_stats['popularity_tier'],
            y=tier_stats['imdb_mean'],
            name='IMDb',
            marker_color='orange',
            opacity=0.7,
            hovertemplate="<b>%{x}</b><br>Avg Rating: %{y:.2f}<br>n=%{customdata}<extra></extra>",
            customdata=tier_stats['count']
        )
    )
    
    # Update layout
    fig.update_layout(
        title="Average Ratings by Movie Popularity Tier",
        xaxis_title="Popularity Tier (IMDb Voters)",
        yaxis_title="Average Rating",
        barmode='group',
        height=500,
        margin=dict(l=60, r=60, t=50, b=100)
    )
    
    # Add count labels
    for i, row in tier_stats.iterrows():
        fig.add_annotation(
            x=row['popularity_tier'],
            y=max(row['b612_mean'], row['imdb_mean']) + 0.2,
            text=f"n={int(row['count'])}",
            showarrow=False,
            font=dict(size=10)
        )
    
    return fig

def create_popularity_difference_figure(analysis_results):
    """Create a figure showing rating differences by popularity tier"""
    tier_stats = analysis_results["tier_stats"]
    
    fig = go.Figure()
    
    # Add bars for rating differences
    colors = ['green' if x >= 0 else 'red' for x in tier_stats['diff_mean']]
    
    fig.add_trace(
        go.Bar(
            x=tier_stats['popularity_tier'],
            y=tier_stats['diff_mean'],
            name='Difference',
            marker_color=colors,
            opacity=0.7,
            error_y=dict(
                type='data',
                array=tier_stats['diff_std'],
                visible=True
            ),
            hovertemplate="<b>%{x}</b><br>Avg Difference: %{y:.2f}<br>±%{error_y.array:.2f}<br>n=%{customdata}<extra></extra>",
            customdata=tier_stats['count']
        )
    )
    
    # Add zero line
    fig.add_shape(
        type="line",
        x0=-0.5,
        y0=0,
        x1=len(tier_stats)-0.5,
        y1=0,
        line=dict(color="black", dash="solid", width=1)
    )
    
    # Add value labels on the bars
    for i, row in tier_stats.iterrows():
        fig.add_annotation(
            x=row['popularity_tier'],
            y=row['diff_mean'] + (0.1 if row['diff_mean'] >= 0 else -0.1),
            text=f"{row['diff_mean']:.2f}",
            showarrow=False,
            font=dict(size=10, color="black"),
            yshift=10 if row['diff_mean'] >= 0 else -10
        )
    
    # Update layout
    fig.update_layout(
        title="Rating Difference by Movie Popularity Tier (Planet B612 - IMDb)",
        xaxis_title="Popularity Tier (IMDb Voters)",
        yaxis_title="Rating Difference",
        height=500,
        margin=dict(l=60, r=60, t=50, b=100)
    )
    
    return fig

def create_niche_vs_mainstream_boxplot(analysis_results):
    """Create a boxplot comparing niche and mainstream movie ratings"""
    df_clean = analysis_results["df_clean"]
    
    fig = go.Figure()
    
    # Create lists of all popularity tiers to compare
    tiers = ['Niche (<10K)', 'Moderate (10K-100K)', 'Popular (100K-500K)', 'Mainstream (>500K)']
    
    # Add box plots
    for tier in tiers:
        tier_data = df_clean[df_clean['popularity_tier'] == tier]['rating_difference']
        
        if not tier_data.empty:
            fig.add_trace(
                go.Box(
                    y=tier_data,
                    name=tier,
                    boxpoints='all',  # show all points
                    jitter=0.3,
                    pointpos=-1.8,
                    marker=dict(size=6, opacity=0.5),
                    hovertemplate="<b>%{text}</b><br>Difference: %{y:.2f}<extra></extra>",
                    text=df_clean[df_clean['popularity_tier'] == tier]['name']
                )
            )
    
    # Add zero line
    fig.add_shape(
        type="line",
        x0=-0.5,
        y0=0,
        x1=len(tiers)-0.5,
        y1=0,
        line=dict(color="black", dash="solid", width=1)
    )
    
    # Update layout
    fig.update_layout(
        title="Distribution of Rating Differences by Popularity Tier",
        xaxis_title="Popularity Tier (IMDb Voters)",
        yaxis_title="Rating Difference (Planet B612 - IMDb)",
        height=600,
        margin=dict(l=60, r=60, t=50, b=60)
    )
    
    return fig

def create_top_movies_comparison(analysis_results):
    """Create a comparison of top rated niche vs. mainstream movies"""
    mainstream_movies = analysis_results["mainstream_top_movies"]
    niche_movies = analysis_results["niche_top_movies"]
    
    # Create subplots with shared y-axis
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Top Niche Movies (<10K voters)", "Top Mainstream Movies (>500K voters)"),
        shared_yaxes=True
    )
    
    # Check if we have niche movies data
    if not niche_movies.empty:
        # Add bars for niche movies
        fig.add_trace(
            go.Bar(
                y=niche_movies['name'],
                x=niche_movies['rating'],
                name='Planet B612',
                marker_color='blue',
                opacity=0.7,
                orientation='h',
                hovertemplate="<b>%{y}</b><br>Planet B612: %{x:.1f}<br>IMDb: %{customdata[0]:.1f}<br>Voters: %{customdata[1]:,.0f}<extra></extra>",
                customdata=np.stack((niche_movies['IMDb'], niche_movies['IMDb_voters']), axis=-1)
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(
                y=niche_movies['name'],
                x=niche_movies['IMDb'],
                name='IMDb',
                marker_color='orange',
                opacity=0.7,
                orientation='h',
                hovertemplate="<b>%{y}</b><br>IMDb: %{x:.1f}<br>Voters: %{customdata:,.0f}<extra></extra>",
                customdata=niche_movies['IMDb_voters']
            ),
            row=1, col=1
        )
    
    # Check if we have mainstream movies data
    if not mainstream_movies.empty:
        # Add bars for mainstream movies
        fig.add_trace(
            go.Bar(
                y=mainstream_movies['name'],
                x=mainstream_movies['rating'],
                name='Planet B612',
                marker_color='blue',
                opacity=0.7,
                orientation='h',
                showlegend=False,
                hovertemplate="<b>%{y}</b><br>Planet B612: %{x:.1f}<br>IMDb: %{customdata[0]:.1f}<br>Voters: %{customdata[1]:,.0f}<extra></extra>",
                customdata=np.stack((mainstream_movies['IMDb'], mainstream_movies['IMDb_voters']), axis=-1)
            ),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Bar(
                y=mainstream_movies['name'],
                x=mainstream_movies['IMDb'],
                name='IMDb',
                marker_color='orange',
                opacity=0.7,
                orientation='h',
                showlegend=False,
                hovertemplate="<b>%{y}</b><br>IMDb: %{x:.1f}<br>Voters: %{customdata:,.0f}<extra></extra>",
                customdata=mainstream_movies['IMDb_voters']
            ),
            row=1, col=2
        )
    
    # Update layout
    fig.update_layout(
        title="Top Rated Movies: Niche vs. Mainstream",
        barmode='group',
        height=700,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=200, r=60, t=80, b=60)
    )
    
    # Update axes
    fig.update_xaxes(title_text="Rating", range=[0, 10], row=1, col=1)
    fig.update_xaxes(title_text="Rating", range=[0, 10], row=1, col=2)
    
    return fig

def create_acclaimed_analysis_figure(analysis_results):
    """Create a chart showing how Planet B612 rates critically acclaimed movies by popularity"""
    df_clean = analysis_results["df_clean"]
    
    # Filter for highly rated IMDb movies (8.0+)
    acclaimed_movies = df_clean[df_clean['IMDb'] >= 8.0].copy()
    
    # Create an empty figure if there are no acclaimed movies
    if acclaimed_movies.empty:
        fig = go.Figure()
        fig.update_layout(
            title="No Critically Acclaimed Movies (IMDb ≥ 8.0) Found in Dataset",
            height=600
        )
        fig.add_annotation(
            x=0.5, y=0.5,
            text="No data available for critically acclaimed movies (IMDb ≥ 8.0)",
            showarrow=False,
            font=dict(size=16)
        )
        return fig
    
    fig = go.Figure()
    
    # Add scatter plot with color based on popularity tier
    fig.add_trace(
        go.Scatter(
            x=acclaimed_movies['IMDb'],
            y=acclaimed_movies['rating'],
            mode='markers',
            marker=dict(
                size=10,
                color=np.log10(acclaimed_movies['IMDb_voters']),
                colorscale='Viridis',
                colorbar=dict(
                    title="IMDb Voters",
                    tickvals=[3, 4, 5, 6, 7],
                    ticktext=['1K', '10K', '100K', '1M', '10M']
                ),
                opacity=0.7
            ),
            text=acclaimed_movies['name'],
            hovertemplate="<b>%{text}</b><br>Planet B612: %{y:.1f}<br>IMDb: %{x:.1f}<br>Voters: %{customdata:,.0f}<extra></extra>",
            customdata=acclaimed_movies['IMDb_voters']
        )
    )
    
    # Add perfect correlation line
    fig.add_trace(
        go.Scatter(
            x=[8, 10],
            y=[8, 10],
            mode='lines',
            line=dict(color='black', dash='dash', width=1),
            name='Perfect Agreement'
        )
    )
    
    # Add regression line
    z = np.polyfit(acclaimed_movies['IMDb'], acclaimed_movies['rating'], 1)
    p = np.poly1d(z)
    x_range = np.linspace(acclaimed_movies['IMDb'].min(), acclaimed_movies['IMDb'].max(), 100)
    
    fig.add_trace(
        go.Scatter(
            x=x_range,
            y=p(x_range),
            mode='lines',
            line=dict(color='red', width=2),
            name=f'Trend Line (y={z[0]:.2f}x+{z[1]:.2f})'
        )
    )
    
    # Calculate correlation
    correlation, p_value = pearsonr(acclaimed_movies['IMDb'], acclaimed_movies['rating'])
    
    # Add correlation annotation
    fig.add_annotation(
        x=0.05,
        y=0.05,
        xref="paper",
        yref="paper",
        text=f"Correlation: {correlation:.3f}<br>p-value: {p_value:.4f}",
        showarrow=False,
        font=dict(size=12),
        bgcolor="white",
        bordercolor="black",
        borderwidth=1,
        borderpad=4
    )
    
    # Update layout
    fig.update_layout(
        title="Planet B612's Ratings of Critically Acclaimed Movies (IMDb ≥ 8.0)",
        xaxis_title="IMDb Rating",
        yaxis_title="Planet B612 Rating",
        height=600,
        margin=dict(l=60, r=60, t=50, b=60)
    )
    
    return fig

def create_popularity_tier_table(analysis_results):
    """Create a table with statistics by popularity tier"""
    tier_stats = analysis_results["tier_stats"]
    t_stat, p_value = analysis_results["niche_vs_mainstream_ttest"]
    
    # Format the table data
    table_data = []
    for _, row in tier_stats.iterrows():
        table_data.append([
            row['popularity_tier'],
            f"{int(row['count'])}",
            f"{row['b612_mean']:.2f}",
            f"{row['imdb_mean']:.2f}",
            f"{row['diff_mean']:.2f}",
            f"{row['diff_std']:.2f}"
        ])
    
    # Create the table
    table = html.Div([
        html.H4("Detailed Statistics by Popularity Tier", className="text-center"),
        html.P(f"T-test between Niche and Mainstream: t={t_stat:.3f}, p-value={p_value:.4f}", 
               className="text-center font-italic") if not pd.isna(t_stat) else html.P(
               "T-test not available: insufficient data in one or both groups",
               className="text-center font-italic"),
        html.P("Statistical significance: " + 
               ("Yes (p<0.05)" if not pd.isna(p_value) and p_value < 0.05 else "No (p≥0.05)"), 
               className="text-center font-weight-bold") if not pd.isna(p_value) else "",
        dbc.Table([
            html.Thead(html.Tr([
                html.Th("Popularity Tier"),
                html.Th("Count"),
                html.Th("Planet B612 Mean"),
                html.Th("IMDb Mean"),
                html.Th("Diff Mean"),
                html.Th("Diff Std")
            ])),
            html.Tbody([
                html.Tr([
                    html.Td(data[0]),
                    html.Td(data[1]),
                    html.Td(data[2]),
                    html.Td(data[3]),
                    html.Td(data[4], style={"color": "green" if float(data[4]) > 0 else "red"}),
                    html.Td(data[5])
                ]) for data in table_data
            ])
        ], bordered=True, hover=True, striped=True, className="mb-4")
    ])
    
    return table

def create_summary_conclusion(analysis_results):
    """Create a summary and conclusion of the popularity bias analysis"""
    correlation = analysis_results["correlation"]
    p_value = analysis_results["p_value"]
    tier_stats = analysis_results["tier_stats"]
    
    # Determine if there's evidence of bias
    has_correlation = abs(correlation) > 0.1 and p_value < 0.05
    
    # Find mainstream and niche differences if they exist
    mainstream_row = tier_stats[tier_stats['popularity_tier'] == 'Mainstream (>500K)']
    niche_row = tier_stats[tier_stats['popularity_tier'] == 'Niche (<10K)']
    
    mainstream_diff = mainstream_row['diff_mean'].values[0] if not mainstream_row.empty else 0
    niche_diff = niche_row['diff_mean'].values[0] if not niche_row.empty else 0
    
    # Determine the main trend
    if correlation < -0.1 and p_value < 0.05:
        trend_text = "There is a statistically significant negative correlation between movie popularity and Planet B612's ratings relative to IMDb. This means that Planet B612 tends to rate more popular movies lower than IMDb does."
    elif correlation > 0.1 and p_value < 0.05:
        trend_text = "There is a statistically significant positive correlation between movie popularity and Planet B612's ratings relative to IMDb. This means that Planet B612 tends to rate more popular movies higher than IMDb does."
    else:
        trend_text = "There is no statistically significant correlation between movie popularity and Planet B612's ratings relative to IMDb."
    
    # Create more detailed conclusions based on available data
    if not mainstream_row.empty and not niche_row.empty:
        if niche_diff > 0 and mainstream_diff < 0:
            conclusion_text = "The data shows that Planet B612 tends to rate niche movies higher than IMDb, while rating mainstream movies lower than IMDb. This supports the allegation that Planet B612 favors niche works over mainstream movies."
        elif niche_diff > mainstream_diff and niche_diff > 0:
            conclusion_text = "The data shows that Planet B612 rates niche movies more favorably (compared to IMDb) than mainstream movies, although the difference may not be extreme."
        elif niche_diff < 0 and mainstream_diff < 0:
            conclusion_text = "Planet B612 tends to be more critical than IMDb for both niche and mainstream movies, but the effect is stronger for mainstream movies."
        elif niche_diff > 0 and mainstream_diff > 0:
            conclusion_text = "Planet B612 tends to rate both niche and mainstream movies higher than IMDb, but the effect is stronger for niche movies."
        else:
            conclusion_text = "The relationship between movie popularity and Planet B612's rating difference is complex and doesn't show a simple bias pattern."
    elif not niche_row.empty:
        if niche_diff > 0:
            conclusion_text = "Planet B612 tends to rate niche movies higher than IMDb, but there are insufficient mainstream movies in the dataset for comparison."
        else:
            conclusion_text = "Planet B612 tends to rate niche movies lower than IMDb, but there are insufficient mainstream movies in the dataset for comparison."
    elif not mainstream_row.empty:
        if mainstream_diff > 0:
            conclusion_text = "Planet B612 tends to rate mainstream movies higher than IMDb, but there are insufficient niche movies in the dataset for comparison."
        else:
            conclusion_text = "Planet B612 tends to rate mainstream movies lower than IMDb, but there are insufficient niche movies in the dataset for comparison."
    else:
        conclusion_text = "There is insufficient data to draw conclusions about Planet B612's rating patterns for different popularity tiers."
    
    # Generate the summary
    summary = html.Div([
        html.H4("Findings and Conclusion", className="text-center"),
        html.Div([
            html.H5("Key Statistics:"),
            html.Ul([
                html.Li([
                    html.Strong("Correlation between popularity and rating difference: "),
                    f"{correlation:.3f} (p-value: {p_value:.4f})"
                ]),
                html.Li([
                    html.Strong("Average rating difference for niche movies (<10K voters): "),
                    html.Span(f"{niche_diff:.2f}", 
                              style={"color": "green" if niche_diff > 0 else "red"})
                ]) if not niche_row.empty else html.Li("No niche movies found in dataset"),
                html.Li([
                    html.Strong("Average rating difference for mainstream movies (>500K voters): "),
                    html.Span(f"{mainstream_diff:.2f}", 
                              style={"color": "green" if mainstream_diff > 0 else "red"})
                ]) if not mainstream_row.empty else html.Li("No mainstream movies found in dataset")
            ]),
            html.H5("Analysis:"),
            html.P(trend_text),
            html.P(conclusion_text),
            html.H5("Context and Interpretation:"),
            html.P([
                "This analysis examines whether Planet B612 shows a systematic bias against mainstream movies, based on the relationship between IMDb voter counts (a proxy for popularity) and rating differences.",
            ]),
            html.P([
                "Film critics often develop specialized tastes that differ from general audience preferences, and may value different aspects of films than casual viewers. Critics may also approach films with deeper analytical frameworks, historical context, and comparison to similar works."
            ]),
            html.P([
                "Whether a preference for niche works represents 'bias' or specialized critical perspective is subjective. Some viewers seek out critics precisely because they highlight overlooked works and apply different standards than audience aggregates."
            ])
        ], className="mt-4")
    ])
    
    return summary

def create_empty_figure(message):
    """Create an empty figure with a message"""
    fig = go.Figure()
    fig.add_annotation(
        x=0.5,
        y=0.5,
        text=message,
        showarrow=False,
        font=dict(size=16)
    )
    fig.update_layout(
        height=400,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    )
    return fig

def create_popularity_bias_tab():
    """Create the layout for the popularity bias tab"""
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H3("Alleged Mainstream Bias Analysis", className="text-center mt-4"),
                html.P("This analysis examines the allegation that Planet B612 favors niche movies over mainstream ones.", className="text-center"),
                html.Hr()
            ], width=12)
        ]),
        
        dbc.Row([
            dbc.Col([
                html.H4("Correlation: Movie Popularity vs. Rating Difference", className="mt-3"),
                dcc.Graph(id='popularity-correlation')
            ], width=12)
        ]),
        
        dbc.Row([
            dbc.Col([
                html.H4("Average Ratings by Popularity Tier", className="mt-3"),
                dcc.Graph(id='popularity-tier-ratings')
            ], width=6),
            dbc.Col([
                html.H4("Rating Differences by Popularity Tier", className="mt-3"),
                dcc.Graph(id='popularity-tier-differences')
            ], width=6)
        ]),
        
        dbc.Row([
            dbc.Col([
                html.H4("Distribution of Rating Differences", className="mt-3"),
                dcc.Graph(id='niche-vs-mainstream-boxplot')
            ], width=12)
        ]),
        
        dbc.Row([
            dbc.Col([
                html.H4("Top Rated Movies: Niche vs. Mainstream", className="mt-3"),
                dcc.Graph(id='top-movies-comparison')
            ], width=12)
        ]),
        
        dbc.Row([
            dbc.Col([
                html.H4("Critically Acclaimed Movies Analysis", className="mt-3"),
                dcc.Graph(id='acclaimed-analysis')
            ], width=12)
        ]),
        
        dbc.Row([
            dbc.Col([
                html.Div(id='popularity-tier-table')
            ], width=12)
        ]),
        
        dbc.Row([
            dbc.Col([
                html.Div(id='popularity-summary')
            ], width=12)
        ]),
        
    ], fluid=True)
def update_popularity_analysis(contents):
    if contents is None:
        empty_fig = create_empty_figure("Upload data to view analysis")
        return empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, None, None
    
    try:
        # Load and clean data
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        df = pd.read_excel(io.BytesIO(decoded))
        
        # Clean column names
        df.columns = [col.strip() if isinstance(col, str) else col for col in df.columns]
        
        # Rename specific columns
        column_mapping = {
            'name ': 'name',
            'year of relase': 'year_of_release',
            'year of review': 'year_of_review',
            'type of review': 'type_of_review',
            'reviewed by': 'reviewer'
        }
        
        df = df.rename(columns=column_mapping)
        
        # Convert IMDb voters to numeric
        if 'IMDb voters' in df.columns:
            # Don't rename or convert here - let analyze_popularity_bias handle it
            # This way we avoid duplicating logic and potential inconsistencies
            pass
        
        # Add rating difference column
        df['rating_difference'] = df['rating'] - df['IMDb']
        
        # Analyze popularity bias
        analysis_results = analyze_popularity_bias(df)
        
        if "error" in analysis_results:
            error_fig = create_empty_figure(analysis_results["error"])
            return error_fig, error_fig, error_fig, error_fig, error_fig, error_fig, None, None
        
        # Create visualizations
        correlation_fig = create_popularity_correlation_figure(analysis_results)
        tier_ratings_fig = create_popularity_tier_figure(analysis_results)
        tier_differences_fig = create_popularity_difference_figure(analysis_results)
        boxplot_fig = create_niche_vs_mainstream_boxplot(analysis_results)
        top_movies_fig = create_top_movies_comparison(analysis_results)
        acclaimed_fig = create_acclaimed_analysis_figure(analysis_results)
        tier_table = create_popularity_tier_table(analysis_results)
        summary = create_summary_conclusion(analysis_results)
        
        return correlation_fig, tier_ratings_fig, tier_differences_fig, boxplot_fig, top_movies_fig, acclaimed_fig, tier_table, summary
        
    except Exception as e:
        error_fig = create_empty_figure(f"Error in analysis: {str(e)}")
        return error_fig, error_fig, error_fig, error_fig, error_fig, error_fig, None, None
    
def add_popularity_bias_tab(app):
    """Add the popularity bias analysis tab to the dashboard"""
    @app.callback(
        Output('popularity-correlation', 'figure'),
        Output('popularity-tier-ratings', 'figure'),
        Output('popularity-tier-differences', 'figure'),
        Output('niche-vs-mainstream-boxplot', 'figure'),
        Output('top-movies-comparison', 'figure'),
        Output('acclaimed-analysis', 'figure'),
        Output('popularity-tier-table', 'children'),
        Output('popularity-summary', 'children'),
        Input('upload-data', 'contents')
    )
    def update_popularity_analysis(contents):
        if contents is None:
            empty_fig = create_empty_figure("Upload data to view analysis")
            return empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, None, None
        
        try:
            # Load and clean data
            content_type, content_string = contents.split(',')
            decoded = base64.b64decode(content_string)
            df = pd.read_excel(io.BytesIO(decoded))
            
            # Clean column names
            df.columns = [col.strip() if isinstance(col, str) else col for col in df.columns]
            
            # Rename specific columns
            column_mapping = {
                'name ': 'name',
                'year of relase': 'year_of_release',
                'year of review': 'year_of_review',
                'type of review': 'type_of_review',
                'reviewed by': 'reviewer'
            }
            
            df = df.rename(columns=column_mapping)
            
            # Add rating difference column
            df['rating_difference'] = df['rating'] - df['IMDb']
            
            # Analyze popularity bias
            analysis_results = analyze_popularity_bias(df)
            
            if "error" in analysis_results:
                error_fig = create_empty_figure(analysis_results["error"])
                return error_fig, error_fig, error_fig, error_fig, error_fig, error_fig, None, None
            
            # Create visualizations
            correlation_fig = create_popularity_correlation_figure(analysis_results)
            tier_ratings_fig = create_popularity_tier_figure(analysis_results)
            tier_differences_fig = create_popularity_difference_figure(analysis_results)
            boxplot_fig = create_niche_vs_mainstream_boxplot(analysis_results)
            top_movies_fig = create_top_movies_comparison(analysis_results)
            acclaimed_fig = create_acclaimed_analysis_figure(analysis_results)
            tier_table = create_popularity_tier_table(analysis_results)
            summary = create_summary_conclusion(analysis_results)
            
            return correlation_fig, tier_ratings_fig, tier_differences_fig, boxplot_fig, top_movies_fig, acclaimed_fig, tier_table, summary
            
        except Exception as e:
            error_fig = create_empty_figure(f"Error in analysis: {str(e)}")
            return error_fig, error_fig, error_fig, error_fig, error_fig, error_fig, None, None    