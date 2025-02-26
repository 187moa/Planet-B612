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
from popularity_bias_analysis import create_popularity_bias_tab, add_popularity_bias_tab

# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

# Define functions to load and process data
def load_and_clean_data(file_content):
    """
    Load and clean the Planet B612 review data from uploaded Excel file
    """
    # Read the Excel file
    content_type, content_string = file_content.split(',')
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

# App layout
app.layout = html.Div([
    dbc.Container([
        # Header
        dbc.Row([
            dbc.Col([
                html.H1("Planet B612 Movie Reviews Dashboard", className="text-center my-4"),
                html.P("An interactive dashboard to explore Planet B612's movie ratings compared to IMDb", className="text-center lead mb-4"),
                html.Hr()
            ], width=12)
        ]),
        
        # File Upload
        dbc.Row([
            dbc.Col([
                dcc.Upload(
                    id='upload-data',
                    children=html.Div([
                        'Drag and Drop or ',
                        html.A('Select the Planet B612 Database Excel File')
                    ]),
                    style={
                        'width': '100%',
                        'height': '60px',
                        'lineHeight': '60px',
                        'borderWidth': '1px',
                        'borderStyle': 'dashed',
                        'borderRadius': '5px',
                        'textAlign': 'center',
                        'margin': '10px'
                    },
                    multiple=False
                ),
                html.Div(id='upload-status', className="text-center")
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

# Store the loaded data
app.config.suppress_callback_exceptions = True

# Load and process data when file is uploaded
@callback(
    Output('upload-status', 'children'),
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
    Input('upload-data', 'contents')
)
def update_dashboard(contents):
    if contents is None:
        return (
            html.P("Please upload the Planet B612 Database Excel file to get started.", className="text-danger"),
            None, # basic-stats
            create_empty_figure("Upload data to view rating distributions"),
            create_empty_figure("Upload data to view rating correlation"),
            create_empty_figure("Upload data to view rating heatmap"),
            create_empty_figure("Upload data to view subcategory analysis"),
            create_empty_figure("Upload data to view subcategory differences"),
            create_empty_figure("Upload data to view release year analysis"),
            create_empty_figure("Upload data to view review year analysis"),
            create_empty_figure("Upload data to view top movies"),
            create_empty_figure("Upload data to view bottom movies"),
            create_empty_figure("Upload data to view most divergent ratings"),
            create_empty_figure("Upload data to view critic profile"),
            None # critical-voice
        )
    
    try:
        # Load and clean data
        df = load_and_clean_data(contents)
        
        # Calculate basic statistics
        stats = calculate_basic_stats(df)
        
        # Create all the visualizations
        return (
            html.P(f"Data loaded successfully! Analyzing {len(df)} movie reviews.", className="text-success"),
            create_basic_stats_card(stats),
            create_rating_distributions(df),
            create_rating_correlation(df),
            create_rating_heatmap(df),
            create_subcategory_analysis(df),
            create_subcategory_differences(df),
            create_release_year_analysis(df),
            create_review_year_analysis(df),
            create_top_movies(df),
            create_bottom_movies(df),
            create_most_divergent(df),
            create_critic_profile(df),
            create_critical_voice_analysis(df)
        )
    except Exception as e:
        return (
            html.P(f"Error loading data: {str(e)}", className="text-danger"),
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

def create_rating_distributions(df):
    """Create distribution plots for Planet B612 and IMDb ratings"""
    fig = make_subplots(rows=3, cols=1, subplot_titles=(
        "Distribution of Planet B612 Ratings",
        "Distribution of IMDb Ratings",
        "Distribution of Rating Differences (Planet B612 - IMDb)"
    ))
    
    # Plot distribution of Planet B612 ratings
    fig.add_trace(
        go.Histogram(
            x=df['rating'],
            nbinsx=20,
            name="Planet B612 Ratings",
            marker_color='blue',
            opacity=0.7
        ),
        row=1, col=1
    )
    
    # Add mean line for Planet B612 ratings
    fig.add_vline(x=df['rating'].mean(), line_dash="dash", line_color="red", row=1, col=1)
    
    # Plot distribution of IMDb ratings
    fig.add_trace(
        go.Histogram(
            x=df['IMDb'],
            nbinsx=20,
            name="IMDb Ratings",
            marker_color='orange',
            opacity=0.7
        ),
        row=2, col=1
    )
    
    # Add mean line for IMDb ratings
    fig.add_vline(x=df['IMDb'].mean(), line_dash="dash", line_color="red", row=2, col=1)
    
    # Plot distribution of rating differences
    fig.add_trace(
        go.Histogram(
            x=df['rating_difference'],
            nbinsx=20,
            name="Rating Differences",
            marker_color='green',
            opacity=0.7
        ),
        row=3, col=1
    )
    
    # Add mean line for differences
    fig.add_vline(x=df['rating_difference'].mean(), line_dash="dash", line_color="red", row=3, col=1)
    
    # Add zero line for differences
    fig.add_vline(x=0, line_dash="dot", line_color="black", row=3, col=1)
    
    # Update layout
    fig.update_layout(
        height=900,
        showlegend=False,
        title_text="Rating Distributions",
    )
    
    fig.update_xaxes(title_text="Rating", row=1, col=1)
    fig.update_xaxes(title_text="Rating", row=2, col=1)
    fig.update_xaxes(title_text="Rating Difference", row=3, col=1)
    
    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=2, col=1)
    fig.update_yaxes(title_text="Count", row=3, col=1)
    
    return fig

def create_rating_correlation(df):
    """Create a scatter plot to show the correlation between Planet B612 and IMDb ratings"""
    # Calculate regression line
    z = np.polyfit(df['IMDb'], df['rating'], 1)
    p = np.poly1d(z)
    
    # Create figure
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
            hovertemplate="<b>%{text}</b><br>Planet B612: %{y:.1f}<br>IMDb: %{x:.1f}<br>Difference: %{marker.color:.1f}<extra></extra>"
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
    x_range = np.linspace(df['IMDb'].min(), df['IMDb'].max(), 100)
    fig.add_trace(
        go.Scatter(
            x=x_range,
            y=p(x_range),
            mode='lines',
            line=dict(color='red', width=2),
            name=f'Regression Line (y={z[0]:.2f}x+{z[1]:.2f})'
        )
    )
    
    # Calculate correlation coefficient
    correlation, p_value = pearsonr(df['IMDb'], df['rating'])
    
    # Add annotation with correlation coefficient
    fig.add_annotation(
        x=0.05,
        y=0.95,
        xref="paper",
        yref="paper",
        text=f"Correlation: {correlation:.2f}<br>p-value: {p_value:.4f}",
        showarrow=False,
        font=dict(size=12),
        bgcolor="white",
        bordercolor="black",
        borderwidth=1,
        borderpad=4
    )
    
    # Add annotations for notable outliers
    top_positive = df.nlargest(3, 'rating_difference')
    top_negative = df.nsmallest(3, 'rating_difference')
    
    for _, row in pd.concat([top_positive, top_negative]).iterrows():
        fig.add_annotation(
            x=row['IMDb'],
            y=row['rating'],
            text=row['name'] if len(row['name']) < 15 else row['name'][:12] + '...',
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=1,
            arrowcolor="black"
        )
    
    # Update layout
    fig.update_layout(
        title="Correlation between Planet B612 and IMDb Ratings",
        xaxis_title="IMDb Rating",
        yaxis_title="Planet B612 Rating",
        showlegend=True,
        height=600,
        legend=dict(x=0.01, y=0.01, bgcolor="white"),
        margin=dict(l=60, r=60, t=50, b=60)
    )
    
    return fig

def create_rating_heatmap(df):
    """Create a heatmap showing the relationship between IMDb ratings and Planet B612 ratings"""
    # Create a 2D histogram / heatmap
    heatmap, xedges, yedges = np.histogram2d(
        df['IMDb'], 
        df['rating'], 
        bins=(10, 10),
        range=[[4, 10], [4, 10]]
    )
    
    # Create z values for heatmap
    z = heatmap.T
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=[(xedges[i] + xedges[i+1])/2 for i in range(len(xedges)-1)],
        y=[(yedges[i] + yedges[i+1])/2 for i in range(len(yedges)-1)],
        colorscale='Blues',
        showscale=True,
        colorbar=dict(title="Number of Movies"),
        hovertemplate="IMDb: %{x:.1f}<br>Planet B612: %{y:.1f}<br>Count: %{z}<extra></extra>"
    ))
    
    # Add a diagonal line for perfect correlation
    fig.add_trace(
        go.Scatter(
            x=[4, 10],
            y=[4, 10],
            mode='lines',
            line=dict(color='red', dash='dash', width=2),
            name='Perfect Agreement'
        )
    )
    
    # Add annotation for the counts in each cell
    for i in range(len(xedges)-1):
        for j in range(len(yedges)-1):
            if heatmap[i, j] > 0:
                fig.add_annotation(
                    x=(xedges[i] + xedges[i+1])/2,
                    y=(yedges[j] + yedges[j+1])/2,
                    text=str(int(heatmap[i, j])),
                    showarrow=False,
                    font=dict(
                        color='white' if heatmap[i, j] > heatmap.max()/2 else 'black'
                    )
                )
    
    # Update layout
    fig.update_layout(
        title="Heatmap of Planet B612 vs IMDb Ratings",
        xaxis_title="IMDb Rating",
        yaxis_title="Planet B612 Rating",
        showlegend=True,
        height=600,
        legend=dict(x=0.01, y=0.99, bgcolor="white"),
        margin=dict(l=60, r=60, t=50, b=60)
    )
    
    return fig

def create_subcategory_analysis(df):
    """Analyze and plot ratings by movie subcategory"""
    if 'subcategory' not in df.columns:
        return create_empty_figure("Subcategory data not found in the uploaded file")
    
    # Group by subcategory and calculate statistics
    subcategory_stats = df.groupby('subcategory').agg({
        'rating': ['count', 'mean'],
        'IMDb': ['mean']
    })
    
    subcategory_stats.columns = ['count', 'b612_mean', 'imdb_mean']
    
    # Filter to only include subcategories with at least 2 movies
    subcategory_stats = subcategory_stats[subcategory_stats['count'] >= 2]
    subcategory_stats = subcategory_stats.sort_values('count', ascending=False)
    subcategory_stats = subcategory_stats.reset_index()
    
    # Create grouped bar chart
    fig = go.Figure()
    
    # Add bars for Planet B612 ratings
    fig.add_trace(
        go.Bar(
            x=subcategory_stats['subcategory'],
            y=subcategory_stats['b612_mean'],
            name='Planet B612',
            marker_color='blue',
            opacity=0.7,
            hovertemplate="%{x}<br>Avg Rating: %{y:.2f}<extra></extra>"
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
            hovertemplate="%{x}<br>Avg Rating: %{y:.2f}<extra></extra>"
        )
    )
    
    # Add annotations for count
    for i, row in subcategory_stats.iterrows():
        fig.add_annotation(
            x=row['subcategory'],
            y=max(row['b612_mean'], row['imdb_mean']) + 0.2,
            text=f"n={int(row['count'])}",
            showarrow=False,
            font=dict(size=10)
        )
    
    # Update layout
    fig.update_layout(
        title="Average Ratings by Movie Subcategory",
        xaxis_title="Subcategory",
        yaxis_title="Average Rating",
        barmode='group',
        xaxis=dict(tickangle=45),
        height=600,
        margin=dict(l=60, r=60, t=50, b=130)  # Increase bottom margin for rotated labels
    )
    
    return fig

def create_subcategory_differences(df):
    """Create a plot of rating differences by subcategory"""
    if 'subcategory' not in df.columns:
        return create_empty_figure("Subcategory data not found in the uploaded file")
    
    # Group by subcategory and calculate statistics
    subcategory_stats = df.groupby('subcategory').agg({
        'rating_difference': ['mean', 'count']
    })
    
    subcategory_stats.columns = ['diff_mean', 'count']
    
    # Filter to only include subcategories with at least 2 movies
    subcategory_stats = subcategory_stats[subcategory_stats['count'] >= 2]
    subcategory_stats = subcategory_stats.sort_values('diff_mean')
    subcategory_stats = subcategory_stats.reset_index()
    
    # Create bar chart
    fig = go.Figure()
    
    # Add bars with colors based on positive/negative difference
    colors = ['red' if x < 0 else 'green' for x in subcategory_stats['diff_mean']]
    
    fig.add_trace(
        go.Bar(
            x=subcategory_stats['subcategory'],
            y=subcategory_stats['diff_mean'],
            marker_color=colors,
            opacity=0.7,
            hovertemplate="%{x}<br>Avg Difference: %{y:.2f}<br>n=%{text}<extra></extra>",
            text=subcategory_stats['count']
        )
    )
    
    # Add zero line
    fig.add_shape(
        type="line",
        x0=-0.5,
        y0=0,
        x1=len(subcategory_stats) - 0.5,
        y1=0,
        line=dict(color="black", width=1, dash="solid"),
    )
    
    # Add value labels on the bars
    for i, row in subcategory_stats.iterrows():
        fig.add_annotation(
            x=row['subcategory'],
            y=row['diff_mean'] + (0.1 if row['diff_mean'] >= 0 else -0.1),
            text=f"{row['diff_mean']:.2f}",
            showarrow=False,
            font=dict(size=10, color="black"),
            yshift=0 if row['diff_mean'] >= 0 else -10
        )
    
    # Update layout
    fig.update_layout(
        title="Rating Difference by Subcategory (Planet B612 - IMDb)",
        xaxis_title="Subcategory",
        yaxis_title="Rating Difference",
        xaxis=dict(tickangle=45),
        height=600,
        margin=dict(l=60, r=60, t=50, b=130)  # Increase bottom margin for rotated labels
    )
    
    return fig

def create_release_year_analysis(df):
    """Analyze how ratings change over release year"""
    if 'year_of_release' not in df.columns:
        return create_empty_figure("Release year data not found in the uploaded file")
    
    # Group by release year
    release_year_stats = df.groupby('year_of_release').agg({
        'rating': ['count', 'mean'],
        'IMDb': ['mean'],
        'rating_difference': ['mean']
    })
    
    release_year_stats.columns = ['count', 'b612_mean', 'imdb_mean', 'diff_mean']
    release_year_stats = release_year_stats.reset_index()
    
    # Only include years with at least 2 movies
    release_year_stats = release_year_stats[release_year_stats['count'] >= 2]
    
    # Create figure with primary and secondary y-axes
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add Planet B612 mean line
    fig.add_trace(
        go.Scatter(
            x=release_year_stats['year_of_release'],
            y=release_year_stats['b612_mean'],
            mode='lines+markers',
            name='Planet B612',
            marker=dict(size=10, color='blue'),
            line=dict(color='blue',width=2),
            hovertemplate="%{x}<br>Avg Rating: %{y:.2f}<extra></extra>"
        ),
        secondary_y=False
    )
    
    # Add IMDb mean line
    fig.add_trace(
        go.Scatter(
            x=release_year_stats['year_of_release'],
            y=release_year_stats['imdb_mean'],
            mode='lines+markers',
            name='IMDb',
            marker=dict(size=10, color='orange'),
            line=dict(color='orange', width=2),
            hovertemplate="%{x}<br>Avg Rating: %{y:.2f}<extra></extra>"
        ),
        secondary_y=False
    )
    
    # Add difference bars
    colors = ['green' if d >= 0 else 'red' for d in release_year_stats['diff_mean']]
    
    fig.add_trace(
        go.Bar(
            x=release_year_stats['year_of_release'],
            y=release_year_stats['diff_mean'],
            name='Difference',
            marker_color=colors,
            opacity=0.3,
            hovertemplate="%{x}<br>Avg Difference: %{y:.2f}<extra></extra>"
        ),
        secondary_y=True
    )
    
    # Add count labels
    for i, row in release_year_stats.iterrows():
        fig.add_annotation(
            x=row['year_of_release'],
            y=max(row['b612_mean'], row['imdb_mean']) + 0.2,
            text=f"n={int(row['count'])}",
            showarrow=False,
            font=dict(size=10)
        )
    
    # Add zero line for difference
    fig.add_shape(
        type="line",
        x0=release_year_stats['year_of_release'].min() - 1,
        y0=0,
        x1=release_year_stats['year_of_release'].max() + 1,
        y1=0,
        line=dict(color="black", width=1, dash="solid"),
        yref="y2"
    )
    
    # Update layout
    fig.update_layout(
        title="Average Ratings by Release Year",
        xaxis_title="Release Year",
        legend=dict(x=0.01, y=0.99, bgcolor="white"),
        height=600,
        margin=dict(l=60, r=60, t=50, b=60)
    )
    
    # Update y-axis labels
    fig.update_yaxes(title_text="Average Rating", secondary_y=False)
    fig.update_yaxes(title_text="Rating Difference", secondary_y=True)
    
    return fig

def create_review_year_analysis(df):
    """Analyze how ratings change over review year"""
    if 'year_of_review' not in df.columns:
        return create_empty_figure("Review year data not found in the uploaded file")
    
    # Group by review year
    review_year_stats = df.groupby('year_of_review').agg({
        'rating': ['count', 'mean'],
        'IMDb': ['mean'],
        'rating_difference': ['mean']
    })
    
    review_year_stats.columns = ['count', 'b612_mean', 'imdb_mean', 'diff_mean']
    review_year_stats = review_year_stats.reset_index()
    
    # Create figure with primary and secondary y-axes
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add Planet B612 mean line
    fig.add_trace(
        go.Scatter(
            x=review_year_stats['year_of_review'],
            y=review_year_stats['b612_mean'],
            mode='lines+markers',
            name='Planet B612',
            marker=dict(size=10, color='blue'),
            line=dict(color='blue', width=2),
            hovertemplate="%{x}<br>Avg Rating: %{y:.2f}<extra></extra>"
        ),
        secondary_y=False
    )
    
    # Add IMDb mean line
    fig.add_trace(
        go.Scatter(
            x=review_year_stats['year_of_review'],
            y=review_year_stats['imdb_mean'],
            mode='lines+markers',
            name='IMDb',
            marker=dict(size=10, color='orange'),
            line=dict(color='orange', width=2),
            hovertemplate="%{x}<br>Avg Rating: %{y:.2f}<extra></extra>"
        ),
        secondary_y=False
    )
    
    # Add difference bars
    colors = ['green' if d >= 0 else 'red' for d in review_year_stats['diff_mean']]
    
    fig.add_trace(
        go.Bar(
            x=review_year_stats['year_of_review'],
            y=review_year_stats['diff_mean'],
            name='Difference',
            marker_color=colors,
            opacity=0.3,
            hovertemplate="%{x}<br>Avg Difference: %{y:.2f}<extra></extra>"
        ),
        secondary_y=True
    )
    
    # Add count labels
    for i, row in review_year_stats.iterrows():
        fig.add_annotation(
            x=row['year_of_review'],
            y=max(row['b612_mean'], row['imdb_mean']) + 0.2,
            text=f"n={int(row['count'])}",
            showarrow=False,
            font=dict(size=10)
        )
    
    # Add zero line for difference
    fig.add_shape(
        type="line",
        x0=review_year_stats['year_of_review'].min() - 1,
        y0=0,
        x1=review_year_stats['year_of_review'].max() + 1,
        y1=0,
        line=dict(color="black", width=1, dash="solid"),
        yref="y2"
    )
    
    # Update layout
    fig.update_layout(
        title="Average Ratings by Review Year",
        xaxis_title="Review Year",
        legend=dict(x=0.01, y=0.99, bgcolor="white"),
        height=600,
        margin=dict(l=60, r=60, t=50, b=60)
    )
    
    # Update y-axis labels
    fig.update_yaxes(title_text="Average Rating", secondary_y=False)
    fig.update_yaxes(title_text="Rating Difference", secondary_y=True)
    
    return fig

def create_top_movies(df):
    """Create visualizations for top rated movies"""
    # Get top 10 movies by Planet B612 rating
    top_movies = df.nlargest(10, 'rating')
    
    # Create figure
    fig = go.Figure()
    
    # Add bars for Planet B612 ratings
    fig.add_trace(
        go.Bar(
            y=top_movies['name'],
            x=top_movies['rating'],
            name='Planet B612',
            marker_color='blue',
            opacity=0.7,
            orientation='h',
            hovertemplate="%{y}<br>Rating: %{x:.1f}<extra></extra>"
        )
    )
    
    # Add bars for IMDb ratings
    fig.add_trace(
        go.Bar(
            y=top_movies['name'],
            x=top_movies['IMDb'],
            name='IMDb',
            marker_color='orange',
            opacity=0.7,
            orientation='h',
            hovertemplate="%{y}<br>Rating: %{x:.1f}<extra></extra>"
        )
    )
    
    # Add rating values as text
    for i, (name, b612, imdb) in enumerate(zip(top_movies['name'], top_movies['rating'], top_movies['IMDb'])):
        # Add Planet B612 rating label
        fig.add_annotation(
            x=b612 + 0.1,
            y=name,
            text=f"{b612:.1f}",
            showarrow=False,
            font=dict(size=10),
            xanchor="left"
        )
        
        # Add IMDb rating label
        fig.add_annotation(
            x=imdb + 0.1,
            y=name,
            text=f"{imdb:.1f}",
            showarrow=False,
            font=dict(size=10),
            xanchor="left"
        )
    
    # Update layout
    fig.update_layout(
        title="Top 10 Movies by Planet B612 Rating",
        xaxis_title="Rating",
        yaxis=dict(autorange="reversed"),  # Reverse y-axis to show highest rating at top
        barmode='group',
        height=600,
        margin=dict(l=200, r=60, t=50, b=60)  # Increase left margin for movie titles
    )
    
    return fig

def create_bottom_movies(df):
    """Create visualizations for bottom rated movies"""
    # Get bottom 10 movies by Planet B612 rating
    bottom_movies = df.nsmallest(10, 'rating')
    
    # Create figure
    fig = go.Figure()
    
    # Add bars for Planet B612 ratings
    fig.add_trace(
        go.Bar(
            y=bottom_movies['name'],
            x=bottom_movies['rating'],
            name='Planet B612',
            marker_color='blue',
            opacity=0.7,
            orientation='h',
            hovertemplate="%{y}<br>Rating: %{x:.1f}<extra></extra>"
        )
    )
    
    # Add bars for IMDb ratings
    fig.add_trace(
        go.Bar(
            y=bottom_movies['name'],
            x=bottom_movies['IMDb'],
            name='IMDb',
            marker_color='orange',
            opacity=0.7,
            orientation='h',
            hovertemplate="%{y}<br>Rating: %{x:.1f}<extra></extra>"
        )
    )
    
    # Add rating values as text
    for i, (name, b612, imdb) in enumerate(zip(bottom_movies['name'], bottom_movies['rating'], bottom_movies['IMDb'])):
        # Add Planet B612 rating label
        fig.add_annotation(
            x=b612 + 0.1,
            y=name,
            text=f"{b612:.1f}",
            showarrow=False,
            font=dict(size=10),
            xanchor="left"
        )
        
        # Add IMDb rating label
        fig.add_annotation(
            x=imdb + 0.1,
            y=name,
            text=f"{imdb:.1f}",
            showarrow=False,
            font=dict(size=10),
            xanchor="left"
        )
    
    # Update layout
    fig.update_layout(
        title="Bottom 10 Movies by Planet B612 Rating",
        xaxis_title="Rating",
        yaxis=dict(autorange="reversed"),  # Reverse y-axis to show lowest rating at top
        barmode='group',
        height=600,
        margin=dict(l=200, r=60, t=50, b=60)  # Increase left margin for movie titles
    )
    
    return fig

def create_most_divergent(df):
    """Plot movies with the most divergent ratings between Planet B612 and IMDb"""
    # Most positive differences (Planet B612 > IMDb)
    most_positive = df.nlargest(5, 'rating_difference')
    
    # Most negative differences (Planet B612 < IMDb)
    most_negative = df.nsmallest(5, 'rating_difference')
    
    # Combine into one dataframe and sort by difference
    most_divergent = pd.concat([most_positive, most_negative])
    most_divergent = most_divergent.sort_values('rating_difference')
    
    # Create figure
    fig = go.Figure()
    
    # Add bars for rating differences
    colors = ['red' if d < 0 else 'green' for d in most_divergent['rating_difference']]
    
    fig.add_trace(
        go.Bar(
            y=most_divergent['name'],
            x=most_divergent['rating_difference'],
            marker_color=colors,
            opacity=0.7,
            orientation='h',
            hovertemplate="%{y}<br>Difference: %{x:.1f}<br>B612: %{customdata[0]:.1f}, IMDb: %{customdata[1]:.1f}<extra></extra>",
            customdata=np.stack((most_divergent['rating'], most_divergent['IMDb']), axis=-1)
        )
    )
    
    # Add zero line
    fig.add_shape(
        type="line",
        x0=0,
        y0=-0.5,
        x1=0,
        y1=len(most_divergent) - 0.5,
        line=dict(color="black", width=1, dash="solid"),
    )
    
    # Add difference values as text
    for i, (name, diff, b612, imdb) in enumerate(zip(
        most_divergent['name'], 
        most_divergent['rating_difference'],
        most_divergent['rating'],
        most_divergent['IMDb']
    )):
        # Add difference label
        fig.add_annotation(
            x=diff + (0.2 if diff >= 0 else -0.2),
            y=name,
            text=f"{diff:.1f} ({b612:.1f} vs {imdb:.1f})",
            showarrow=False,
            font=dict(size=10),
            xanchor="left" if diff >= 0 else "right"
        )
    
    # Update layout
    fig.update_layout(
        title="Movies with Most Divergent Ratings (Planet B612 - IMDb)",
        xaxis_title="Rating Difference",
        height=600,
        margin=dict(l=200, r=120, t=50, b=60)  # Increase margins for labels
    )
    
    return fig

def create_critic_profile(df):
    """Create a radar chart to visualize Planet B612's critic profile compared to IMDb"""
    if 'subcategory' not in df.columns:
        return create_empty_figure("Subcategory data not found in the uploaded file")
    
    # Get the top 6 subcategories by movie count
    top_subcategories = df['subcategory'].value_counts().nlargest(6).index.tolist()
    
    # Calculate mean ratings for these subcategories
    subcategory_data = df[df['subcategory'].isin(top_subcategories)].groupby('subcategory').agg({
        'rating': 'mean',
        'IMDb': 'mean'
    }).reindex(top_subcategories)
    
    # Create data for the radar chart
    categories = subcategory_data.index.tolist()
    b612_values = subcategory_data['rating'].tolist()
    imdb_values = subcategory_data['IMDb'].tolist()
    
    # Add the first category to the end to close the loop
    categories = categories + [categories[0]]
    b612_values = b612_values + [b612_values[0]]
    imdb_values = imdb_values + [imdb_values[0]]
    
    # Create radar chart
    fig = go.Figure()
    
    # Add Planet B612 trace
    fig.add_trace(
        go.Scatterpolar(
            r=b612_values,
            theta=categories,
            fill='toself',
            name='Planet B612',
            line_color='blue',
            fillcolor='rgba(0, 0, 255, 0.1)'
        )
    )
    
    # Add IMDb trace
    fig.add_trace(
        go.Scatterpolar(
            r=imdb_values,
            theta=categories,
            fill='toself',
            name='IMDb',
            line_color='orange',
            fillcolor='rgba(255, 165, 0, 0.1)'
        )
    )
    
    # Update layout
    fig.update_layout(
        title="Planet B612 vs IMDb: Genre Rating Profile",
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[5, 9]
            )
        ),
        showlegend=True,
        height=600
    )
    
    return fig

def create_critical_voice_analysis(df):
    """Analyze Planet B612's critical voice and philosophy"""
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
                ], width=6)
            ]),
            dbc.Row([
                dbc.Col([
                    html.H5("Critical Style", className="mt-4"),
                    html.P(get_critic_style_description(higher_percent, lower_percent, align_percent))
                ], width=12)
            ])
        ])
    ], className="mb-4")

def get_critic_style_description(higher_percent, lower_percent, align_percent):
    """Determine the critic style based on rating patterns"""
    if lower_percent > higher_percent + 20:
        return "Planet B612 demonstrates a primarily critical voice, often challenging popular consensus by providing more stringent evaluations."
    elif higher_percent > lower_percent + 20:
        return "Planet B612 demonstrates a primarily appreciative voice, often championing films beyond their popular reception."
    elif align_percent > 60:
        return "Planet B612 demonstrates a primarily consensual voice, generally aligning with broader audience sentiment while providing nuanced context."
    else:
        return "Planet B612 demonstrates a balanced critical voice, alternating between praise and criticism based on the specific merits of each work."

# Add the popularity bias tab callback to register it with the app
add_popularity_bias_tab(app)

# Run the app
if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 8050))
    app.run_server(debug=False, host='0.0.0.0', port=port)