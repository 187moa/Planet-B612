import dash
from dash import dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import pearsonr
import io
import base64
import os
from popularity_bias_analysis import (
    create_popularity_bias_tab,  # Already imported
    add_popularity_bias_tab,     # Already imported
    analyze_popularity_bias,     # New import
    create_popularity_correlation_figure,  # New import
    create_popularity_tier_figure,  # New import
    create_popularity_difference_figure,  # New import
    create_niche_vs_mainstream_boxplot,  # New import
    create_top_movies_comparison,  # New import
    create_acclaimed_analysis_figure,  # New import
    create_popularity_tier_table,  # New import
    create_summary_conclusion  # New import
)
from planet_b612_analysis import (
    plot_rating_distributions,
    plot_rating_correlation,
    create_heatmap_comparison,
    plot_subcategory_analysis,
    plot_timeline_analysis,
    plot_top_movies,
    plot_most_divergent,
    create_critic_profile,
    
)

# If create_critical_voice_analysis is in the planet_b612_insights.py file:


# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

# Global variable to store processed data
processed_data = None

# Define functions to load and process data
def load_and_clean_data(file_path):
    """
    Load and clean the Planet B612 review data from Excel file
    """
    # Read the Excel file directly
    df = pd.read_excel(file_path)
    
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
        df['IMDb_voters'] = df['IMDb voters'].astype(str).str.replace(',', '').astype(float)
        df = df.drop('IMDb voters', axis=1)
    
    # Add rating difference column
    df['rating_difference'] = df['rating'] - df['IMDb']
    
    # Drop any unnecessary columns
    columns_to_drop = [col for col in df.columns if col.startswith('__EMPTY')]
    df = df.drop(columns_to_drop, axis=1, errors='ignore')
    
    # Clean up category values
    if 'category' in df.columns:
        df['category'] = df['category'].str.strip()
    
    if 'subcategory' in df.columns:
        df['subcategory'] = df['subcategory'].str.strip()
    
    # Convert years to integers where possible
    for col in ['year_of_release', 'year_of_review']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df

def calculate_basic_stats(df):
    """
    Calculate basic statistics for Planet B612 and IMDb ratings
    """
    # Basic stats for Planet B612 ratings
    b612_stats = df['rating'].describe()
    imdb_stats = df['IMDb'].describe()
    diff_stats = df['rating_difference'].describe()
    
    # Calculate correlation between Planet B612 and IMDb ratings
    correlation, p_value = pearsonr(df['rating'], df['IMDb'])
    
    return {
        'b612': b612_stats,
        'imdb': imdb_stats,
        'diff': diff_stats,
        'correlation': correlation,
        'p_value': p_value
    }

def load_data():
    """Load and process data at app startup"""
    global processed_data
    
    try:
        # Path to the Excel file in the repository
        # Use os.path.join for cross-platform compatibility
        file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Planet B612 Database .xlsx')
        
        # Load and clean the data
        df = load_and_clean_data(file_path)
        
        # Calculate basic statistics
        stats = calculate_basic_stats(df)
        
        # Store the processed data for reuse
        processed_data = {
            'df': df,
            'stats': stats
        }
        
        print(f"Successfully loaded data with {len(df)} records")
        return True
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return False

# App layout without upload component
app.layout = html.Div([
    dbc.Container([
        # Header
        dbc.Row([
            dbc.Col([
                html.H1("Planet B612 Movie Reviews Dashboard", className="text-center my-4"),
                html.P("An interactive dashboard exploring Planet B612's movie ratings compared to IMDb", className="text-center lead mb-4"),
                html.Hr()
            ], width=12)
        ]),
        
        # Status indicator
        dbc.Row([
            dbc.Col([
                html.Div(id='data-status')
            ], width=12)
        ]),
        
        # Main Tabs
        dbc.Row([
            dbc.Col([
                dcc.Tabs(id='tabs', value='overview', children=[
                    dcc.Tab(label='Overview', value='overview', children=[
                        dbc.Row([
                            dbc.Col([
                                html.H3("Basic Statistics", className="text-center mt-4"),
                                html.Div(id='basic-stats')
                            ], width=12)
                        ])
                    ]),
                    dcc.Tab(label='Rating Comparison', value='rating-comparison', children=[
                        dbc.Row([
                            dbc.Col([
                                html.H3("Rating Distributions", className="mt-4"),
                                dcc.Graph(id='rating-distributions')
                            ], width=12),
                            dbc.Col([
                                html.H3("Rating Correlation", className="mt-4"),
                                dcc.Graph(id='rating-correlation')
                            ], width=12),
                            dbc.Col([
                                html.H3("Rating Heatmap", className="mt-4"),
                                dcc.Graph(id='rating-heatmap')
                            ], width=12)
                        ])
                    ]),
                    dcc.Tab(label='Genre Analysis', value='genre-analysis', children=[
                        dbc.Row([
                            dbc.Col([
                                html.H3("Ratings by Subcategory", className="mt-4"),
                                dcc.Graph(id='subcategory-analysis')
                            ], width=12),
                            dbc.Col([
                                html.H3("Rating Differences by Subcategory", className="mt-4"),
                                dcc.Graph(id='subcategory-differences')
                            ], width=12)
                        ])
                    ]),
                    dcc.Tab(label='Timeline Analysis', value='timeline-analysis', children=[
                        dbc.Row([
                            dbc.Col([
                                html.H3("Ratings by Release Year", className="mt-4"),
                                dcc.Graph(id='release-year-analysis')
                            ], width=12),
                            dbc.Col([
                                html.H3("Ratings by Review Year", className="mt-4"),
                                dcc.Graph(id='review-year-analysis')
                            ], width=12)
                        ])
                    ]),
                    dcc.Tab(label='Notable Movies', value='notable-movies', children=[
                        dbc.Row([
                            dbc.Col([
                                html.H3("Top Rated Movies", className="mt-4"),
                                dcc.Graph(id='top-movies')
                            ], width=12),
                            dbc.Col([
                                html.H3("Bottom Rated Movies", className="mt-4"),
                                dcc.Graph(id='bottom-movies')
                            ], width=12),
                            dbc.Col([
                                html.H3("Most Divergent Ratings", className="mt-4"),
                                dcc.Graph(id='most-divergent')
                            ], width=12)
                        ])
                    ]),
                    dcc.Tab(label='Critic Identity', value='critic-identity', children=[
                        dbc.Row([
                            dbc.Col([
                                html.H3("Critic Profile", className="mt-4"),
                                dcc.Graph(id='critic-profile')
                            ], width=12),
                            dbc.Col([
                                html.H3("Critical Voice Analysis", className="mt-4"),
                                html.Div(id='critical-voice')
                            ], width=12)
                        ])
                    ]),
                    dcc.Tab(label='Popularity Bias Analysis', value='popularity-bias', children=[
                        create_popularity_bias_tab()
                    ])
                ])
            ], width=12)
        ])
    ], fluid=True)
])
# Add these functions to planet_b612_dashboard.py, before your callback

def create_rating_distributions(df):
    """Create rating distributions visualization using Plotly"""
    fig = make_subplots(rows=3, cols=1, 
                      subplot_titles=("Distribution of Planet B612 Ratings", 
                                     "Distribution of IMDb Ratings",
                                     "Distribution of Rating Differences"))
    
    # Plot distribution of Planet B612 ratings
    fig.add_trace(
        go.Histogram(x=df['rating'], nbinsx=20, name="Planet B612",
                    marker_color='blue', opacity=0.7),
        row=1, col=1
    )
    
    # Add mean line for Planet B612
    fig.add_vline(x=df['rating'].mean(), line_dash="dash", line_color="red", row=1, col=1)
    fig.add_annotation(
        x=df['rating'].mean() + 0.1, y=0.9, 
        text=f'Mean: {df["rating"].mean():.2f}',
        showarrow=False, font=dict(color="red"),
        xref="x", yref="paper",
        row=1, col=1
    )
    
    # Plot distribution of IMDb ratings
    fig.add_trace(
        go.Histogram(x=df['IMDb'], nbinsx=20, name="IMDb",
                   marker_color='orange', opacity=0.7),
        row=2, col=1
    )
    
    # Add mean line for IMDb
    fig.add_vline(x=df['IMDb'].mean(), line_dash="dash", line_color="red", row=2, col=1)
    fig.add_annotation(
        x=df['IMDb'].mean() + 0.1, y=0.9, 
        text=f'Mean: {df["IMDb"].mean():.2f}',
        showarrow=False, font=dict(color="red"),
        xref="x2", yref="paper",
        row=2, col=1
    )
    
    # Plot distribution of rating differences
    fig.add_trace(
        go.Histogram(x=df['rating_difference'], nbinsx=20, name="Difference",
                   marker_color='green', opacity=0.7),
        row=3, col=1
    )
    
    # Add mean line for differences
    fig.add_vline(x=df['rating_difference'].mean(), line_dash="dash", line_color="red", row=3, col=1)
    fig.add_vline(x=0, line_dash="dot", line_color="black", row=3, col=1)
    fig.add_annotation(
        x=df['rating_difference'].mean() + 0.1, y=0.9, 
        text=f'Mean: {df["rating_difference"].mean():.2f}',
        showarrow=False, font=dict(color="red"),
        xref="x3", yref="paper",
        row=3, col=1
    )
    
    fig.update_layout(height=800, showlegend=False)
    return fig

def create_rating_correlation(df):
    """Create rating correlation visualization using Plotly"""
    # Calculate correlation
    correlation, p_value = pearsonr(df['IMDb'], df['rating'])
    
    # Create regression line data
    z = np.polyfit(df['IMDb'], df['rating'], 1)
    p = np.poly1d(z)
    x_range = np.linspace(df['IMDb'].min(), df['IMDb'].max(), 100)
    
    fig = go.Figure()
    
    # Add scatter plot
    fig.add_trace(
        go.Scatter(
            x=df['IMDb'],
            y=df['rating'],
            mode='markers',
            marker=dict(
                size=10,
                color=df['rating_difference'],
                colorscale='RdBu_r',
                colorbar=dict(title="Rating Difference"),
                opacity=0.7
            ),
            text=df['name'],
            hovertemplate="<b>%{text}</b><br>IMDb: %{x:.1f}<br>Planet B612: %{y:.1f}<br>Diff: %{marker.color:.1f}<extra></extra>"
        )
    )
    
    # Add perfect correlation line
    min_val = min(df['rating'].min(), df['IMDb'].min())
    max_val = max(df['rating'].max(), df['IMDb'].max())
    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            line=dict(color='black', dash='dash', width=1),
            name='Perfect Agreement'
        )
    )
    
    # Add regression line
    fig.add_trace(
        go.Scatter(
            x=x_range,
            y=p(x_range),
            mode='lines',
            line=dict(color='red', width=2),
            name=f'Regression Line (y={z[0]:.2f}x+{z[1]:.2f})'
        )
    )
    
    # Add correlation annotation
    fig.add_annotation(
        x=0.05, y=0.95,
        xref="paper", yref="paper",
        text=f"Correlation: {correlation:.2f}<br>p-value: {p_value:.4f}",
        showarrow=False,
        font=dict(size=12),
        bgcolor="white",
        bordercolor="black",
        borderwidth=1,
        borderpad=4
    )
    
    # Annotate notable outliers
    # Find the 5 most positive and 5 most negative differences
    top_positive = df.nlargest(5, 'rating_difference')
    top_negative = df.nsmallest(5, 'rating_difference')
    
    for _, row in pd.concat([top_positive, top_negative]).iterrows():
        movie_name = row['name'].strip()
        if len(movie_name) > 15:
            movie_name = movie_name[:12] + '...'
            
        fig.add_annotation(
            x=row['IMDb'],
            y=row['rating'],
            text=movie_name,
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=1,
            arrowcolor="black"
        )
    
    fig.update_layout(
        title="Correlation between Planet B612 and IMDb Ratings",
        xaxis_title="IMDb Rating",
        yaxis_title="Planet B612 Rating",
        height=600,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def create_heatmap_comparison(df):
    """Create rating heatmap visualization using Plotly"""
    # Create a 2D histogram
    heatmap, xedges, yedges = np.histogram2d(
        df['IMDb'], 
        df['rating'], 
        bins=(10, 10),
        range=[[4, 10], [4, 10]]
    )
    
    # Create a heatmap using Plotly
    fig = go.Figure(data=go.Heatmap(
        z=heatmap.T,
        x=[f"{x:.1f}" for x in np.linspace(4, 10, 10)],
        y=[f"{y:.1f}" for y in np.linspace(4, 10, 10)],
        colorscale='Blues',
        showscale=True,
        colorbar=dict(title="Number of Movies"),
        hovertemplate="IMDb: %{x}<br>Planet B612: %{y}<br>Count: %{z}<extra></extra>"
    ))
    
    # Add a diagonal line for perfect correlation
    fig.add_trace(
        go.Scatter(
            x=[4, 10],
            y=[4, 10],
            mode='lines',
            line=dict(color='red', dash='dash', width=1),
            name='Perfect Agreement'
        )
    )
    
    # Add value annotations
    for i in range(len(heatmap)):
        for j in range(len(heatmap[i])):
            if heatmap[i, j] > 0:
                fig.add_annotation(
                    x=xedges[i] + (xedges[i+1] - xedges[i])/2,
                    y=yedges[j] + (yedges[j+1] - yedges[j])/2,
                    text=str(int(heatmap[i, j])),
                    showarrow=False,
                    font=dict(color="white" if heatmap[i, j] > heatmap.max()/2 else "black")
                )
    
    fig.update_layout(
        title="Heatmap of Planet B612 vs IMDb Ratings",
        xaxis_title="IMDb Rating",
        yaxis_title="Planet B612 Rating",
        height=600
    )
    
    return fig

def create_subcategory_analysis(df):
    """Create subcategory analysis visualization using Plotly"""
    # Group by subcategory and calculate statistics
    subcategory_stats = df.groupby('subcategory').agg({
        'rating': ['count', 'mean'],
        'IMDb': ['mean'],
        'rating_difference': ['mean']
    })
    
    subcategory_stats.columns = ['count', 'b612_mean', 'imdb_mean', 'diff_mean']
    subcategory_stats = subcategory_stats.sort_values('count', ascending=False)
    
    # Filter to only include subcategories with at least 2 movies
    subcategory_stats = subcategory_stats[subcategory_stats['count'] >= 2]
    subcategory_stats = subcategory_stats.reset_index()
    
    fig = go.Figure()
    
    # Add bars for Planet B612 ratings
    fig.add_trace(
        go.Bar(
            x=subcategory_stats['subcategory'],
            y=subcategory_stats['b612_mean'],
            name='Planet B612',
            marker_color='blue',
            opacity=0.7,
            hovertemplate="<b>%{x}</b><br>Rating: %{y:.2f}<br>Count: %{customdata}<extra></extra>",
            customdata=subcategory_stats['count']
        )
    )
    
    # Add bars for IMDb ratings
    fig.add_trace(
        go.Bar(
            x=subcategory_stats['subcategory'],
            y=subcategory_stats['imdb_mean'],
            name='IMDb',
            marker_color='orange',
            opacity=0.7,
            hovertemplate="<b>%{x}</b><br>Rating: %{y:.2f}<br>Count: %{customdata}<extra></extra>",
            customdata=subcategory_stats['count']
        )
    )
    
    # Add count labels
    for i, row in subcategory_stats.iterrows():
        fig.add_annotation(
            x=row['subcategory'],
            y=max(row['b612_mean'], row['imdb_mean']) + 0.2,
            text=f"n={int(row['count'])}",
            showarrow=False,
            font=dict(size=10)
        )
    
    fig.update_layout(
        title="Average Ratings by Movie Subcategory",
        xaxis_title="Subcategory",
        yaxis_title="Average Rating",
        xaxis_tickangle=-45,
        height=600,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        barmode='group'
    )
    
    return fig

def create_subcategory_differences(df):
    """Create subcategory differences visualization using Plotly"""
    # Group by subcategory and calculate statistics
    subcategory_stats = df.groupby('subcategory').agg({
        'rating': ['count', 'mean'],
        'IMDb': ['mean'],
        'rating_difference': ['mean', 'std']
    })
    
    subcategory_stats.columns = ['count', 'b612_mean', 'imdb_mean', 'diff_mean', 'diff_std']
    subcategory_stats = subcategory_stats.sort_values('diff_mean')
    
    # Filter to only include subcategories with at least 2 movies
    subcategory_stats = subcategory_stats[subcategory_stats['count'] >= 2]
    subcategory_stats = subcategory_stats.reset_index()
    
    fig = go.Figure()
    
    # Add bars for rating differences
    colors = ['green' if x >= 0 else 'red' for x in subcategory_stats['diff_mean']]
    
    fig.add_trace(
        go.Bar(
            x=subcategory_stats['subcategory'],
            y=subcategory_stats['diff_mean'],
            marker_color=colors,
            opacity=0.7,
            error_y=dict(
                type='data',
                array=subcategory_stats['diff_std'],
                visible=True
            ),
            hovertemplate="<b>%{x}</b><br>Difference: %{y:.2f}<br>±%{error_y.array:.2f}<br>Count: %{customdata}<extra></extra>",
            customdata=subcategory_stats['count']
        )
    )
    
    # Add zero line
    fig.add_shape(
        type="line",
        x0=-0.5,
        y0=0,
        x1=len(subcategory_stats)-0.5,
        y1=0,
        line=dict(color="black", dash="solid", width=1)
    )
    
    # Add value labels
    for i, row in subcategory_stats.iterrows():
        fig.add_annotation(
            x=row['subcategory'],
            y=row['diff_mean'] + (0.1 if row['diff_mean'] >= 0 else -0.1),
            text=f"{row['diff_mean']:.2f}",
            showarrow=False,
            font=dict(size=10, color="black")
        )
    
    fig.update_layout(
        title="Rating Difference by Subcategory (Planet B612 - IMDb)",
        xaxis_title="Subcategory",
        yaxis_title="Rating Difference",
        xaxis_tickangle=-45,
        height=600
    )
    
    return fig

def create_timeline_analysis(df, by_release_year=True):
    """Create timeline analysis visualization using Plotly"""
    # Group by year
    if by_release_year:
        year_col = 'year_of_release'
        title = "Average Ratings by Release Year"
    else:
        year_col = 'year_of_review'
        title = "Average Ratings by Review Year"
    
    # Skip if column doesn't exist
    if year_col not in df.columns:
        return create_empty_figure(f"No {year_col} data available")
    
    # Group by year and calculate statistics
    year_stats = df.groupby(year_col).agg({
        'rating': ['count', 'mean'],
        'IMDb': ['mean'],
        'rating_difference': ['mean']
    })
    
    year_stats.columns = ['count', 'b612_mean', 'imdb_mean', 'diff_mean']
    
    # Only include years with at least 2 movies
    year_stats = year_stats[year_stats['count'] >= 2]
    year_stats = year_stats.reset_index()
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add lines for Planet B612 and IMDb ratings
    fig.add_trace(
        go.Scatter(
            x=year_stats[year_col],
            y=year_stats['b612_mean'],
            mode='lines+markers',
            name='Planet B612',
            line=dict(color='blue', width=2),
            marker=dict(size=8),
            hovertemplate="<b>%{x}</b><br>Planet B612: %{y:.2f}<br>Count: %{customdata}<extra></extra>",
            customdata=year_stats['count']
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=year_stats[year_col],
            y=year_stats['imdb_mean'],
            mode='lines+markers',
            name='IMDb',
            line=dict(color='orange', width=2),
            marker=dict(size=8),
            hovertemplate="<b>%{x}</b><br>IMDb: %{y:.2f}<br>Count: %{customdata}<extra></extra>",
            customdata=year_stats['count']
        )
    )
    
    # Add the difference as a bar plot on secondary y-axis
    fig.add_trace(
        go.Bar(
            x=year_stats[year_col],
            y=year_stats['diff_mean'],
            name='Difference',
            marker_color=['green' if x >= 0 else 'red' for x in year_stats['diff_mean']],
            opacity=0.3,
            hovertemplate="<b>%{x}</b><br>Difference: %{y:.2f}<extra></extra>"
        ),
        secondary_y=True
    )
    
    # Add count labels
    for i, row in year_stats.iterrows():
        fig.add_annotation(
            x=row[year_col],
            y=max(row['b612_mean'], row['imdb_mean']) + 0.2,
            text=f"n={int(row['count'])}",
            showarrow=False,
            font=dict(size=10)
        )
    
    # Add zero line for difference
    fig.add_shape(
        type="line",
        x0=year_stats[year_col].min() - 1,
        y0=0,
        x1=year_stats[year_col].max() + 1,
        y1=0,
        line=dict(color="black", dash="dot", width=1),
        yref="y2"
    )
    
    fig.update_layout(
        title=title,
        xaxis_title="Year",
        yaxis_title="Average Rating",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        height=600
    )
    
    fig.update_yaxes(title_text="Rating Difference", secondary_y=True)
    
    return fig

def create_top_movies(df, top=True):
    """Create top or bottom movies visualization using Plotly"""
    if top:
        movies = df.nlargest(10, 'rating')
        title = "Top 10 Movies by Planet B612 Rating"
    else:
        movies = df.nsmallest(10, 'rating')
        title = "Bottom 10 Movies by Planet B612 Rating"
    
    fig = go.Figure()
    
    # Add bars for Planet B612 ratings
    fig.add_trace(
        go.Bar(
            y=movies['name'],
            x=movies['rating'],
            name='Planet B612',
            marker_color='blue',
            opacity=0.7,
            orientation='h',
            hovertemplate="<b>%{y}</b><br>Planet B612: %{x:.1f}<extra></extra>"
        )
    )
    
    # Add bars for IMDb ratings
    fig.add_trace(
        go.Bar(
            y=movies['name'],
            x=movies['IMDb'],
            name='IMDb',
            marker_color='orange',
            opacity=0.7,
            orientation='h',
            hovertemplate="<b>%{y}</b><br>IMDb: %{x:.1f}<extra></extra>"
        )
    )
    
    # Add rating values as text
    for i, (b612, imdb) in enumerate(zip(movies['rating'], movies['IMDb'])):
        fig.add_annotation(
            x=b612 + 0.1,
            y=i,
            text=f"{b612:.1f}",
            showarrow=False,
            font=dict(color="black", size=10),
            xanchor="left"
        )
        
        fig.add_annotation(
            x=imdb + 0.1,
            y=i,
            text=f"{imdb:.1f}",
            showarrow=False,
            font=dict(color="black", size=10),
            xanchor="left"
        )
    
    fig.update_layout(
        title=title,
        xaxis_title="Rating",
        xaxis=dict(range=[0, 10.5]),
        height=600,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        barmode='group'
    )
    
    return fig

def create_most_divergent(df):
    """Create most divergent movies visualization using Plotly"""
    # Most positive differences (Planet B612 > IMDb)
    most_positive = df.nlargest(10, 'rating_difference')
    
    # Most negative differences (Planet B612 < IMDb)
    most_negative = df.nsmallest(10, 'rating_difference')
    
    fig = make_subplots(rows=2, cols=1, 
                       subplot_titles=("Movies Where Planet B612 Rates Higher Than IMDb",
                                      "Movies Where Planet B612 Rates Lower Than IMDb"),
                       vertical_spacing=0.1)
    
    # Add bars for most positive differences
    fig.add_trace(
        go.Bar(
            y=most_positive['name'],
            x=most_positive['rating_difference'],
            marker_color='green',
            opacity=0.7,
            orientation='h',
            hovertemplate="<b>%{y}</b><br>B612: %{customdata[0]:.1f}<br>IMDb: %{customdata[1]:.1f}<br>Diff: +%{x:.1f}<extra></extra>",
            customdata=np.stack((most_positive['rating'], most_positive['IMDb']), axis=-1)
        ),
        row=1, col=1
    )
    
    # Add bars for most negative differences
    fig.add_trace(
        go.Bar(
            y=most_negative['name'],
            x=most_negative['rating_difference'],
            marker_color='red',
            opacity=0.7,
            orientation='h',
            hovertemplate="<b>%{y}</b><br>B612: %{customdata[0]:.1f}<br>IMDb: %{customdata[1]:.1f}<br>Diff: %{x:.1f}<extra></extra>",
            customdata=np.stack((most_negative['rating'], most_negative['IMDb']), axis=-1)
        ),
        row=2, col=1
    )
    
    # Add value labels for positive differences
    for i, row in enumerate(most_positive['rating_difference']):
        fig.add_annotation(
            x=row + 0.05,
            y=i,
            text=f"+{row:.1f}",
            showarrow=False,
            font=dict(size=10),
            xref="x", yref="y"
        )
    
    # Add value labels for negative differences
    for i, row in enumerate(most_negative['rating_difference']):
        fig.add_annotation(
            x=row - 0.05,
            y=i,
            text=f"{row:.1f}",
            showarrow=False,
            font=dict(size=10),
            xanchor="right",
            xref="x2", yref="y2"
        )
    
    # Add zero lines
    fig.add_shape(
        type="line", 
        x0=0, y0=-0.5, 
        x1=0, y1=len(most_positive)-0.5,
        line=dict(color="black", dash="solid", width=1),
        row=1, col=1
    )
    
    fig.add_shape(
        type="line", 
        x0=0, y0=-0.5, 
        x1=0, y1=len(most_negative)-0.5,
        line=dict(color="black", dash="solid", width=1),
        row=2, col=1
    )
    
    fig.update_layout(
        height=800,
        showlegend=False
    )
    
    return fig

def create_critic_profile(df):
    """Create critic profile visualization using Plotly"""
    # Get the top 6 subcategories by movie count
    top_subcategories = df['subcategory'].value_counts().nlargest(6).index.tolist()
    
    # Calculate mean ratings for these subcategories
    subcategory_data = df[df['subcategory'].isin(top_subcategories)].groupby('subcategory').agg({
        'rating': 'mean',
        'IMDb': 'mean'
    }).reindex(top_subcategories)
    
    # Prepare data for radar chart
    categories = subcategory_data.index.tolist()
    
    fig = go.Figure()
    
    # Add Planet B612 trace
    fig.add_trace(go.Scatterpolar(
        r=subcategory_data['rating'].tolist(),
        theta=categories,
        fill='toself',
        name='Planet B612',
        line_color='blue',
        fillcolor='rgba(0, 0, 255, 0.1)'
    ))
    
    # Add IMDb trace
    fig.add_trace(go.Scatterpolar(
        r=subcategory_data['IMDb'].tolist(),
        theta=categories,
        fill='toself',
        name='IMDb',
        line_color='orange',
        fillcolor='rgba(255, 165, 0, 0.1)'
    ))
    
    # Update layout
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[5, 9]
            )
        ),
        title="Planet B612 vs IMDb: Genre Rating Profile",
        height=600,
        showlegend=True
    )
    
    return fig
# Initialize the data when the app starts
load_data()

# Modified callback that uses pre-loaded data
# Replace your existing update_dashboard callback with this one:

@callback(
    Output('data-status', 'children'),
    Output('basic-stats', 'children'),
    Output('rating-distributions', 'figure'),
    Output('rating-correlation', 'figure'),
    Output('rating-heatmap', 'figure'),
    Output('subcategory-analysis', 'figure'),
    Output('subcategory-differences', 'figure'),
    Output('release-year-analysis', 'figure'),
    Output('review-year-analysis', 'figure'),
    Output('top-movies', 'figure'),
    Output('bottom-movies', 'figure'),
    Output('most-divergent', 'figure'),
    Output('critic-profile', 'figure'),
    Output('critical-voice', 'children'),
    Input('tabs', 'value')  # Only depends on tab selection now
)
def update_dashboard(tab_value):
    global processed_data
    
    if processed_data is None:
        # Try loading data again if it failed initially
        success = load_data()
        if not success:
            return (
                html.P("Error: Could not load Planet B612 Database. Please check if the file exists.", className="text-danger"),
                None, # basic-stats
                create_empty_figure("Error loading data"),
                create_empty_figure("Error loading data"),
                create_empty_figure("Error loading data"),
                create_empty_figure("Error loading data"),
                create_empty_figure("Error loading data"),
                create_empty_figure("Error loading data"),
                create_empty_figure("Error loading data"),
                create_empty_figure("Error loading data"),
                create_empty_figure("Error loading data"),
                create_empty_figure("Error loading data"),
                create_empty_figure("Error loading data"),
                None # critical-voice
            )
    
    # Use the pre-loaded data
    df = processed_data['df']
    stats = processed_data['stats']
    
    # Create all the visualizations using the new Plotly-based functions
    return (
        html.P(f"Data loaded successfully! Analyzing {len(df)} movie reviews.", className="text-success"),
        create_basic_stats_card(stats),
        create_rating_distributions(df),
        create_rating_correlation(df),
        create_heatmap_comparison(df),
        create_subcategory_analysis(df),
        create_subcategory_differences(df),
        create_timeline_analysis(df, by_release_year=True),
        create_timeline_analysis(df, by_release_year=False),
        create_top_movies(df, top=True),
        create_top_movies(df, top=False),
        create_most_divergent(df),
        create_critic_profile(df),
        create_critical_voice_analysis(df)
    )

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

def create_basic_stats_card(stats):
    """Create a card with basic statistics"""
    if stats is None:
        return None
    
    b612_stats = stats['b612']
    imdb_stats = stats['imdb']
    diff_stats = stats['diff']
    correlation = stats['correlation']
    p_value = stats['p_value']
    
    return dbc.Card([
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.H4("Planet B612 Ratings", className="card-title"),
                    html.P(f"Mean: {b612_stats['mean']:.2f}"),
                    html.P(f"Median: {b612_stats['50%']:.2f}"),
                    html.P(f"Min: {b612_stats['min']:.2f}"),
                    html.P(f"Max: {b612_stats['max']:.2f}"),
                    html.P(f"Std Dev: {b612_stats['std']:.2f}")
                ], width=4),
                dbc.Col([
                    html.H4("IMDb Ratings", className="card-title"),
                    html.P(f"Mean: {imdb_stats['mean']:.2f}"),
                    html.P(f"Median: {imdb_stats['50%']:.2f}"),
                    html.P(f"Min: {imdb_stats['min']:.2f}"),
                    html.P(f"Max: {imdb_stats['max']:.2f}"),
                    html.P(f"Std Dev: {imdb_stats['std']:.2f}")
                ], width=4),
                dbc.Col([
                    html.H4("Rating Differences", className="card-title"),
                    html.P(f"Mean Difference: {diff_stats['mean']:.2f}"),
                    html.P(f"Median Difference: {diff_stats['50%']:.2f}"),
                    html.P(f"Min Difference: {diff_stats['min']:.2f}"),
                    html.P(f"Max Difference: {diff_stats['max']:.2f}"),
                    html.P(f"Correlation: {correlation:.2f} (p-value: {p_value:.4f})")
                ], width=4)
            ])
        ])
    ], className="mb-4")

# The rest of your visualization functions remain unchanged...
# (create_rating_distributions, create_rating_correlation, etc.)

# Keep all your existing visualization creation functions...

# Modified callback for the popularity bias tab
def update_popularity_analysis_callback():
    global processed_data
    
    @callback(
        Output('popularity-correlation', 'figure'),
        Output('popularity-tier-ratings', 'figure'),
        Output('popularity-tier-differences', 'figure'),
        Output('niche-vs-mainstream-boxplot', 'figure'),
        Output('top-movies-comparison', 'figure'),
        Output('acclaimed-analysis', 'figure'),
        Output('popularity-tier-table', 'children'),
        Output('popularity-summary', 'children'),
        Input('tabs', 'value')  # Only depends on tab selection now
    )
    def update_popularity_analysis(tab_value):
        if processed_data is None or 'df' not in processed_data:
            empty_fig = create_empty_figure("Error loading data")
            return empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, None, None
        
        try:
            # Use the pre-loaded data
            df = processed_data['df']
            
            # Only process if we're on the popularity bias tab
            if tab_value != 'popularity-bias':
                empty_fig = create_empty_figure("Select the Popularity Bias tab to view analysis")
                return empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, None, None
            
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
def create_critical_voice_analysis(df):
    """
    Analyze Planet B612's critical voice and philosophy based on rating patterns
    Returns a formatted card with analysis results
    """
    # Calculate how often Planet B612 aligns with, exceeds, or falls below IMDb
    align_threshold = 0.5  # Within 0.5 points is considered "aligned"
    
    aligned_count = sum(abs(df['rating_difference']) <= align_threshold)
    higher_count = sum(df['rating_difference'] > align_threshold)
    lower_count = sum(df['rating_difference'] < -align_threshold)
    
    align_percent = aligned_count / len(df) * 100
    higher_percent = higher_count / len(df) * 100
    lower_percent = lower_count / len(df) * 100
    
    # Calculate the "dissent index" - how often Planet B612 disagrees with IMDb consensus
    df['significant_difference'] = abs(df['rating_difference']) > 1.0
    dissent_index = df['significant_difference'].mean() * 100
    
    # Calculate "harshness index" - how often Planet B612 rates lower than IMDb
    harshness_index = (df['rating_difference'] < 0).mean() * 100
    
    # Calculate "enthusiasm index" - average rating for top rated movies
    enthusiasm_index = df.nlargest(10, 'rating')['rating'].mean()
    
    # Calculate how Planet B612 treats popular vs. niche films
    if 'IMDb_voters' in df.columns:
        # Define popularity threshold (e.g., more than 100,000 votes)
        df['popular'] = df['IMDb_voters'] > 100000
        popular_diff = df[df['popular']]['rating_difference'].mean()
        niche_diff = df[~df['popular']]['rating_difference'].mean()
        popularity_bias = popular_diff - niche_diff
    else:
        popularity_bias = None
    
    # Determine the critic style based on rating patterns
    if lower_percent > higher_percent + 20:
        critic_style = "Primarily critical voice, often challenging popular consensus by providing more stringent evaluations"
    elif higher_percent > lower_percent + 20:
        critic_style = "Primarily appreciative voice, often championing films beyond their popular reception"
    elif align_percent > 60:
        critic_style = "Primarily consensual voice, generally aligning with broader audience sentiment while providing nuanced context"
    else:
        critic_style = "Balanced critical voice, alternating between praise and criticism based on the specific merits of each work"
    
    # Create the analysis card
    return dbc.Card([
        dbc.CardHeader(html.H4("Planet B612's Critical Voice Analysis")),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.H5("Rating Tendencies"),
                    html.P([
                        html.Strong("Alignment with IMDb: "), 
                        f"{align_percent:.1f}% of ratings are within 0.5 points of IMDb"
                    ]),
                    html.P([
                        html.Strong("Higher than IMDb: "), 
                        f"{higher_percent:.1f}% of ratings exceed IMDb by more than 0.5 points"
                    ]),
                    html.P([
                        html.Strong("Lower than IMDb: "), 
                        f"{lower_percent:.1f}% of ratings fall below IMDb by more than 0.5 points"
                    ]),
                ], width=6),
                dbc.Col([
                    html.H5("Critical Identity"),
                    html.P([
                        html.Strong("Dissent Index: "), 
                        f"{dissent_index:.1f}% of reviews differ significantly from IMDb consensus"
                    ]),
                    html.P([
                        html.Strong("Harshness Index: "), 
                        f"Planet B612 rates movies lower than IMDb {harshness_index:.1f}% of the time"
                    ]),
                    html.P([
                        html.Strong("Enthusiasm Level: "), 
                        f"Average rating for top 10 movies is {enthusiasm_index:.2f}/10"
                    ]),
                    html.P([
                        html.Strong("Popularity Bias: "), 
                        f"Difference of {popularity_bias:.2f} points between popular and niche films" if popularity_bias is not None else "No popularity data available"
                    ]),
                ], width=6)
            ]),
            dbc.Row([
                dbc.Col([
                    html.H5("Critical Style", className="mt-4"),
                    html.P(critic_style),
                    html.Hr(),
                    html.H5("Genre Preferences"),
                    html.Div(id="genre-preferences", children=create_genre_preference_analysis(df))
                ], width=12)
            ])
        ])
    ], className="mb-4")

def create_genre_preference_analysis(df):
    """Helper function to analyze genre preferences"""
    if 'subcategory' not in df.columns:
        return html.P("Subcategory data not available for genre analysis")
    
    # Calculate average rating difference by subcategory
    subcategory_diffs = df.groupby('subcategory')['rating_difference'].agg(['mean', 'count'])
    subcategory_diffs = subcategory_diffs[subcategory_diffs['count'] >= 2]  # At least 2 movies
    
    # Find favorites and least favorites
    if subcategory_diffs.empty:
        return html.P("Not enough genre data for meaningful analysis")
        
    favorites = subcategory_diffs.nlargest(3, 'mean')
    least_favorites = subcategory_diffs.nsmallest(3, 'mean')
    
    return html.Div([
        html.P("Genres where Planet B612 is most generous compared to IMDb:"),
        html.Ul([
            html.Li(f"{genre}: +{row['mean']:.2f} points higher (based on {row['count']} films)") 
            for genre, row in favorites.iterrows() if row['mean'] > 0
        ]),
        html.P("Genres where Planet B612 is most critical compared to IMDb:"),
        html.Ul([
            html.Li(f"{genre}: {row['mean']:.2f} points lower (based on {row['count']} films)") 
            for genre, row in least_favorites.iterrows() if row['mean'] < 0
        ])
    ])
# Register the popularity bias callback
update_popularity_analysis_callback()

# Make sure that app.config.suppress_callback_exceptions is set
app.config.suppress_callback_exceptions = True
