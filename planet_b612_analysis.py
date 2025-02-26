import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec

# Set style for better visualizations
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("deep")
sns.set_context("talk")

def load_and_clean_data(file_path):
    """
    Load and clean the Planet B612 review data from Excel file
    """
    # Read the Excel file
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
    df['category'] = df['category'].str.strip()
    
    # Convert years to integers where possible
    for col in ['year_of_release', 'year_of_review']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df

def calculate_basic_stats(df):
    """
    Calculate and print basic statistics for Planet B612 and IMDb ratings
    """
    print("\n========== BASIC STATISTICS ==========")
    
    # Basic stats for Planet B612 ratings
    b612_stats = df['rating'].describe()
    imdb_stats = df['IMDb'].describe()
    diff_stats = df['rating_difference'].describe()
    
    print("\nPlanet B612 Rating Statistics:")
    print(f"Count: {b612_stats['count']}")
    print(f"Mean: {b612_stats['mean']:.2f}")
    print(f"Median: {df['rating'].median():.2f}")
    print(f"Min: {b612_stats['min']:.2f}")
    print(f"Max: {b612_stats['max']:.2f}")
    print(f"Standard Deviation: {b612_stats['std']:.2f}")
    
    print("\nIMDb Rating Statistics:")
    print(f"Count: {imdb_stats['count']}")
    print(f"Mean: {imdb_stats['mean']:.2f}")
    print(f"Median: {df['IMDb'].median():.2f}")
    print(f"Min: {imdb_stats['min']:.2f}")
    print(f"Max: {imdb_stats['max']:.2f}")
    print(f"Standard Deviation: {imdb_stats['std']:.2f}")
    
    print("\nRating Difference Statistics (Planet B612 - IMDb):")
    print(f"Mean Difference: {diff_stats['mean']:.2f}")
    print(f"Median Difference: {df['rating_difference'].median():.2f}")
    print(f"Min Difference: {diff_stats['min']:.2f}")
    print(f"Max Difference: {diff_stats['max']:.2f}")
    print(f"Standard Deviation: {diff_stats['std']:.2f}")
    
    # Calculate correlation between Planet B612 and IMDb ratings
    correlation, p_value = pearsonr(df['rating'], df['IMDb'])
    print(f"\nCorrelation between Planet B612 and IMDb ratings: {correlation:.2f} (p-value: {p_value:.4f})")

def plot_rating_distributions(df):
    """
    Create distribution plots for Planet B612 and IMDb ratings
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 15))
    
    # Plot distribution of Planet B612 ratings
    sns.histplot(df['rating'], bins=20, kde=True, color='blue', ax=axes[0])
    axes[0].axvline(df['rating'].mean(), color='red', linestyle='dashed', linewidth=2)
    axes[0].text(
        df['rating'].mean() + 0.1, 
        axes[0].get_ylim()[1] * 0.9, 
        f'Mean: {df["rating"].mean():.2f}', 
        color='red'
    )
    axes[0].set_title('Distribution of Planet B612 Ratings', fontsize=16)
    axes[0].set_xlabel('Rating', fontsize=14)
    axes[0].set_ylabel('Count', fontsize=14)
    
    # Plot distribution of IMDb ratings
    sns.histplot(df['IMDb'], bins=20, kde=True, color='orange', ax=axes[1])
    axes[1].axvline(df['IMDb'].mean(), color='red', linestyle='dashed', linewidth=2)
    axes[1].text(
        df['IMDb'].mean() + 0.1, 
        axes[1].get_ylim()[1] * 0.9, 
        f'Mean: {df["IMDb"].mean():.2f}', 
        color='red'
    )
    axes[1].set_title('Distribution of IMDb Ratings', fontsize=16)
    axes[1].set_xlabel('Rating', fontsize=14)
    axes[1].set_ylabel('Count', fontsize=14)
    
    # Plot distribution of rating differences
    sns.histplot(df['rating_difference'], bins=20, kde=True, color='green', ax=axes[2])
    axes[2].axvline(df['rating_difference'].mean(), color='red', linestyle='dashed', linewidth=2)
    axes[2].text(
        df['rating_difference'].mean() + 0.1, 
        axes[2].get_ylim()[1] * 0.9, 
        f'Mean: {df["rating_difference"].mean():.2f}', 
        color='red'
    )
    axes[2].axvline(0, color='black', linestyle='dotted', linewidth=1)
    axes[2].set_title('Distribution of Rating Differences (Planet B612 - IMDb)', fontsize=16)
    axes[2].set_xlabel('Rating Difference', fontsize=14)
    axes[2].set_ylabel('Count', fontsize=14)
    
    plt.tight_layout()
    plt.savefig('rating_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_rating_correlation(df):
    """
    Create a scatter plot to show the correlation between Planet B612 and IMDb ratings
    """
    plt.figure(figsize=(10, 10))
    
    # Create scatter plot
    scatter = plt.scatter(
        df['IMDb'], 
        df['rating'], 
        alpha=0.7, 
        c=df['rating_difference'],
        cmap='RdBu_r',
        s=100
    )
    
    # Add perfect correlation line
    min_val = min(df['rating'].min(), df['IMDb'].min())
    max_val = max(df['rating'].max(), df['IMDb'].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='Perfect Agreement')
    
    # Add regression line
    z = np.polyfit(df['IMDb'], df['rating'], 1)
    p = np.poly1d(z)
    plt.plot(np.sort(df['IMDb']), p(np.sort(df['IMDb'])), "r-", alpha=0.7, 
             label=f'Regression Line (y={z[0]:.2f}x+{z[1]:.2f})')
    
    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Rating Difference (Planet B612 - IMDb)', fontsize=12)
    
    # Add movie titles for notable outliers
    # Find the 5 most positive and 5 most negative differences
    top_positive = df.nlargest(5, 'rating_difference')
    top_negative = df.nsmallest(5, 'rating_difference')
    
    # Add annotations for these points
    for _, row in pd.concat([top_positive, top_negative]).iterrows():
        plt.annotate(
            row['name'].strip() if len(row['name'].strip()) < 15 else row['name'].strip()[:12] + '...',
            xy=(row['IMDb'], row['rating']),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=8,
            bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7)
        )
    
    # Calculate and show correlation coefficient
    correlation, p_value = pearsonr(df['IMDb'], df['rating'])
    plt.annotate(
        f'Correlation: {correlation:.2f}\np-value: {p_value:.4f}',
        xy=(0.05, 0.95),
        xycoords='axes fraction',
        fontsize=12,
        bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.7)
    )
    
    plt.title('Correlation between Planet B612 and IMDb Ratings', fontsize=16)
    plt.xlabel('IMDb Rating', fontsize=14)
    plt.ylabel('Planet B612 Rating', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='lower right')
    
    plt.tight_layout()
    plt.savefig('rating_correlation.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_subcategory_analysis(df):
    """
    Analyze and plot ratings by movie subcategory
    """
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
    
    # Create figure with two subplots
    fig, axes = plt.subplots(2, 1, figsize=(14, 16))
    
    # Plot the mean ratings by subcategory
    subcategory_stats = subcategory_stats.reset_index()
    
    # Melt the dataframe for easier plotting
    melted_df = pd.melt(
        subcategory_stats,
        id_vars=['subcategory', 'count'],
        value_vars=['b612_mean', 'imdb_mean'],
        var_name='rating_source',
        value_name='mean_rating'
    )
    
    # Create the grouped bar chart
    bar_plot = sns.barplot(
        x='subcategory', 
        y='mean_rating', 
        hue='rating_source', 
        data=melted_df,
        ax=axes[0],
        palette=['blue', 'orange']
    )
    
    # Add count labels above bars
    for i, count in enumerate(subcategory_stats['count']):
        axes[0].text(
            i, 
            subcategory_stats['b612_mean'].max() + 0.2, 
            f'n={count}',
            ha='center',
            va='bottom',
            fontsize=9
        )
    
    # Customize the first plot
    axes[0].set_title('Average Ratings by Movie Subcategory', fontsize=16)
    axes[0].set_xlabel('Subcategory', fontsize=14)
    axes[0].set_ylabel('Average Rating', fontsize=14)
    axes[0].set_ylim(0, subcategory_stats['b612_mean'].max() + 1)
    axes[0].legend(labels=['Planet B612', 'IMDb'])
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45, ha='right')
    
    # Plot the mean differences by subcategory
    diff_df = subcategory_stats.sort_values('diff_mean')
    
    bars = axes[1].bar(
        diff_df['subcategory'],
        diff_df['diff_mean'],
        color=[
            'red' if x < 0 else 'green' for x in diff_df['diff_mean']
        ]
    )
    
    # Add value labels on the bars
    for bar in bars:
        height = bar.get_height()
        label_height = height + 0.05 if height > 0 else height - 0.2
        axes[1].text(
            bar.get_x() + bar.get_width()/2., 
            label_height,
            f'{height:.2f}',
            ha='center', 
            va='bottom' if height > 0 else 'top',
            fontsize=9
        )
    
    # Customize the second plot
    axes[1].set_title('Rating Difference by Subcategory (Planet B612 - IMDb)', fontsize=16)
    axes[1].set_xlabel('Subcategory', fontsize=14)
    axes[1].set_ylabel('Rating Difference', fontsize=14)
    axes[1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45, ha='right')
    
    # Add a legend for the second plot
    red_patch = mpatches.Patch(color='red', label='Planet B612 < IMDb')
    green_patch = mpatches.Patch(color='green', label='Planet B612 > IMDb')
    axes[1].legend(handles=[red_patch, green_patch], loc='lower right')
    
    plt.tight_layout()
    plt.savefig('subcategory_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_timeline_analysis(df):
    """
    Analyze how ratings change over time (both release year and review year)
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 16))
    
    # Plot by release year
    release_year_stats = df.groupby('year_of_release').agg({
        'rating': ['count', 'mean'],
        'IMDb': ['mean'],
        'rating_difference': ['mean']
    })
    
    release_year_stats.columns = ['count', 'b612_mean', 'imdb_mean', 'diff_mean']
    release_year_stats = release_year_stats.reset_index()
    
    # Only include years with at least 2 movies
    release_year_stats = release_year_stats[release_year_stats['count'] >= 2]
    
    # Plot mean ratings by release year
    axes[0].plot(release_year_stats['year_of_release'], release_year_stats['b612_mean'], 
                marker='o', linestyle='-', linewidth=2, label='Planet B612')
    axes[0].plot(release_year_stats['year_of_release'], release_year_stats['imdb_mean'], 
                marker='s', linestyle='-', linewidth=2, label='IMDb')
    
    # Add count labels
    for i, row in release_year_stats.iterrows():
        axes[0].text(
            row['year_of_release'], 
            max(row['b612_mean'], row['imdb_mean']) + 0.1, 
            f'n={int(row["count"])}',
            ha='center',
            fontsize=8
        )
    
    # Add the difference as a bar plot at the bottom
    ax2 = axes[0].twinx()
    ax2.bar(
        release_year_stats['year_of_release'], 
        release_year_stats['diff_mean'], 
        alpha=0.2, 
        color=['red' if x < 0 else 'green' for x in release_year_stats['diff_mean']],
        width=0.8
    )
    ax2.set_ylabel('Rating Difference', fontsize=12)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Customize the first plot
    axes[0].set_title('Average Ratings by Release Year', fontsize=16)
    axes[0].set_xlabel('Release Year', fontsize=14)
    axes[0].set_ylabel('Average Rating', fontsize=14)
    axes[0].legend(loc='upper left')
    axes[0].grid(True, alpha=0.3)
    
    # Plot by review year
    review_year_stats = df.groupby('year_of_review').agg({
        'rating': ['count', 'mean'],
        'IMDb': ['mean'],
        'rating_difference': ['mean']
    })
    
    review_year_stats.columns = ['count', 'b612_mean', 'imdb_mean', 'diff_mean']
    review_year_stats = review_year_stats.reset_index()
    
    # Plot mean ratings by review year
    axes[1].plot(review_year_stats['year_of_review'], review_year_stats['b612_mean'], 
                marker='o', linestyle='-', linewidth=2, label='Planet B612')
    axes[1].plot(review_year_stats['year_of_review'], review_year_stats['imdb_mean'], 
                marker='s', linestyle='-', linewidth=2, label='IMDb')
    
    # Add count labels
    for i, row in review_year_stats.iterrows():
        axes[1].text(
            row['year_of_review'], 
            max(row['b612_mean'], row['imdb_mean']) + 0.1, 
            f'n={int(row["count"])}',
            ha='center',
            fontsize=8
        )
    
    # Add the difference as a bar plot at the bottom
    ax3 = axes[1].twinx()
    ax3.bar(
        review_year_stats['year_of_review'], 
        review_year_stats['diff_mean'], 
        alpha=0.2, 
        color=['red' if x < 0 else 'green' for x in review_year_stats['diff_mean']],
        width=0.5
    )
    ax3.set_ylabel('Rating Difference', fontsize=12)
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Customize the second plot
    axes[1].set_title('Average Ratings by Review Year', fontsize=16)
    axes[1].set_xlabel('Review Year', fontsize=14)
    axes[1].set_ylabel('Average Rating', fontsize=14)
    axes[1].legend(loc='upper left')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('timeline_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_heatmap_comparison(df):
    """
    Create a heatmap showing the relationship between IMDb ratings and Planet B612 ratings
    """
    plt.figure(figsize=(12, 10))
    
    # Create a 2D histogram / heatmap
    heatmap, xedges, yedges = np.histogram2d(
        df['IMDb'], 
        df['rating'], 
        bins=(10, 10),
        range=[[4, 10], [4, 10]]
    )
    
    # Create a custom colormap from white to deep blue
    colors = [(1, 1, 1), (0, 0, 0.5)]  # White to deep blue
    cmap = LinearSegmentedColormap.from_list("custom_cmap", colors, N=100)
    
    # Plot the heatmap
    plt.imshow(
        heatmap.T, 
        origin='lower', 
        aspect='auto',
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        cmap=cmap
    )
    
    plt.colorbar(label='Number of Movies')
    
    # Add a diagonal line for perfect correlation
    plt.plot([4, 10], [4, 10], 'r--', alpha=0.7, label='Perfect Agreement')
    
    # Add annotation for the counts in each cell
    for i in range(len(xedges)-1):
        for j in range(len(yedges)-1):
            if heatmap[i, j] > 0:
                plt.text(
                    xedges[i] + (xedges[i+1] - xedges[i])/2,
                    yedges[j] + (yedges[j+1] - yedges[j])/2,
                    str(int(heatmap[i, j])),
                    ha='center',
                    va='center',
                    color='white' if heatmap[i, j] > heatmap.max()/2 else 'black'
                )
    
    plt.title('Heatmap of Planet B612 vs IMDb Ratings', fontsize=16)
    plt.xlabel('IMDb Rating', fontsize=14)
    plt.ylabel('Planet B612 Rating', fontsize=14)
    plt.grid(False)
    plt.legend(loc='upper left')
    
    plt.tight_layout()
    plt.savefig('rating_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_top_movies(df):
    """
    Create visualizations for top and bottom rated movies
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 16))
    
    # Get top 10 movies by Planet B612 rating
    top_movies = df.nlargest(10, 'rating')
    
    # Create a horizontal bar chart for top movies
    y_pos = range(len(top_movies))
    
    # First bar chart for top movies
    axes[0].barh(y_pos, top_movies['rating'], height=0.4, align='center', 
                color='blue', alpha=0.6, label='Planet B612')
    axes[0].barh(y_pos, top_movies['IMDb'], height=0.4, align='edge', 
                color='orange', alpha=0.6, label='IMDb')
    
    # Add movie names as y-axis labels
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels([name[:30] + '...' if len(name) > 30 else name 
                             for name in top_movies['name']])
    
    # Add rating values as text
    for i, (b612, imdb) in enumerate(zip(top_movies['rating'], top_movies['IMDb'])):
        axes[0].text(b612 + 0.1, i - 0.1, f'{b612:.1f}', va='center', fontsize=9)
        axes[0].text(imdb + 0.1, i + 0.1, f'{imdb:.1f}', va='center', fontsize=9)
    
    # Customize the first plot
    axes[0].set_title('Top 10 Movies by Planet B612 Rating', fontsize=16)
    axes[0].set_xlabel('Rating', fontsize=14)
    axes[0].legend(loc='lower right')
    axes[0].set_xlim(0, 10.5)  # Adjust as needed based on your data
    
    # Get bottom 10 movies by Planet B612 rating
    bottom_movies = df.nsmallest(10, 'rating')
    
    # Second bar chart for bottom movies
    axes[1].barh(range(len(bottom_movies)), bottom_movies['rating'], height=0.4, align='center', 
                color='blue', alpha=0.6, label='Planet B612')
    axes[1].barh(range(len(bottom_movies)), bottom_movies['IMDb'], height=0.4, align='edge', 
                color='orange', alpha=0.6, label='IMDb')
    
    # Add movie names as y-axis labels
    axes[1].set_yticks(range(len(bottom_movies)))
    axes[1].set_yticklabels([name[:30] + '...' if len(name) > 30 else name 
                             for name in bottom_movies['name']])
    
    # Add rating values as text
    for i, (b612, imdb) in enumerate(zip(bottom_movies['rating'], bottom_movies['IMDb'])):
        axes[1].text(b612 + 0.1, i - 0.1, f'{b612:.1f}', va='center', fontsize=9)
        axes[1].text(imdb + 0.1, i + 0.1, f'{imdb:.1f}', va='center', fontsize=9)
    
    # Customize the second plot
    axes[1].set_title('Bottom 10 Movies by Planet B612 Rating', fontsize=16)
    axes[1].set_xlabel('Rating', fontsize=14)
    axes[1].legend(loc='lower right')
    axes[1].set_xlim(0, 10.5)  # Adjust as needed based on your data
    
    plt.tight_layout()
    plt.savefig('top_bottom_movies.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_most_divergent(df):
    """
    Plot movies with the most divergent ratings between Planet B612 and IMDb
    """
    # Most positive differences (Planet B612 > IMDb)
    most_positive = df.nlargest(10, 'rating_difference')
    
    # Most negative differences (Planet B612 < IMDb)
    most_negative = df.nsmallest(10, 'rating_difference')
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 16))
    
    # Plot most positive differences
    axes[0].barh(range(len(most_positive)), most_positive['rating_difference'], 
                color='green', alpha=0.7)
    
    # Add movie names
    axes[0].set_yticks(range(len(most_positive)))
    axes[0].set_yticklabels([f"{name[:25]}... ({b612:.1f} vs {imdb:.1f})" 
                            if len(name) > 25 else f"{name} ({b612:.1f} vs {imdb:.1f})" 
                            for name, b612, imdb in zip(
                                most_positive['name'],
                                most_positive['rating'],
                                most_positive['IMDb']
                            )])
    
    # Add value labels
    for i, diff in enumerate(most_positive['rating_difference']):
        axes[0].text(diff + 0.05, i, f'+{diff:.1f}', va='center')
    
    axes[0].set_title('Movies Where Planet B612 Rates Higher Than IMDb', fontsize=16)
    axes[0].set_xlabel('Rating Difference (Planet B612 - IMDb)', fontsize=14)
    axes[0].axvline(x=0, color='black', linestyle='-', alpha=0.3)
    
    # Plot most negative differences
    axes[1].barh(range(len(most_negative)), most_negative['rating_difference'], 
                color='red', alpha=0.7)
    
    # Add movie names
    axes[1].set_yticks(range(len(most_negative)))
    axes[1].set_yticklabels([f"{name[:25]}... ({b612:.1f} vs {imdb:.1f})" 
                            if len(name) > 25 else f"{name} ({b612:.1f} vs {imdb:.1f})" 
                            for name, b612, imdb in zip(
                                most_negative['name'],
                                most_negative['rating'],
                                most_negative['IMDb']
                            )])
    
    # Add value labels
    for i, diff in enumerate(most_negative['rating_difference']):
        axes[1].text(diff - 0.2, i, f'{diff:.1f}', va='center')
    
    axes[1].set_title('Movies Where Planet B612 Rates Lower Than IMDb', fontsize=16)
    axes[1].set_xlabel('Rating Difference (Planet B612 - IMDb)', fontsize=14)
    axes[1].axvline(x=0, color='black', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('most_divergent.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_dashboard(df):
    """
    Create a comprehensive dashboard of all analyses
    """
    plt.figure(figsize=(22, 28))
    gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 1])
    
    # Add a main title
    plt.suptitle('Planet B612 Movie Critics Analysis Dashboard', fontsize=24, y=0.995)
    
    # 1. Rating Correlation Plot
    ax1 = plt.subplot(gs[0, 0])
    
    # Create scatter plot
    scatter = ax1.scatter(
        df['IMDb'], 
        df['rating'], 
        alpha=0.7, 
        c=df['rating_difference'],
        cmap='RdBu_r',
        s=80
    )
    
    # Add perfect correlation line
    min_val = min(df['rating'].min(), df['IMDb'].min())
    max_val = max(df['rating'].max(), df['IMDb'].max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='Perfect Agreement')
    
    # Add regression line
    z = np.polyfit(df['IMDb'], df['rating'], 1)
    p = np.poly1d(z)
    ax1.plot(np.sort(df['IMDb']), p(np.sort(df['IMDb'])), "r-", alpha=0.7,
            label=f'Regression Line (y={z[0]:.2f}x+{z[1]:.2f})')
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax1)
    cbar.set_label('Rating Difference', fontsize=10)
    
    # Calculate and show correlation coefficient
    correlation, p_value = pearsonr(df['IMDb'], df['rating'])
    ax1.annotate(
        f'Correlation: {correlation:.2f}\np-value: {p_value:.4f}',
        xy=(0.05, 0.95),
        xycoords='axes fraction',
        fontsize=10,
        bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.7)
    )
    
    ax1.set_title('Correlation: Planet B612 vs IMDb Ratings', fontsize=14)
    ax1.set_xlabel('IMDb Rating', fontsize=12)
    ax1.set_ylabel('Planet B612 Rating', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='lower right', fontsize=8)
    
    # 2. Heatmap Comparison
    ax2 = plt.subplot(gs[0, 1])
    
    # Create a 2D histogram / heatmap
    heatmap, xedges, yedges = np.histogram2d(
        df['IMDb'], 
        df['rating'], 
        bins=(8, 8),
        range=[[4, 10], [3, 10]]
    )
    
    # Create a custom colormap from white to deep blue
    colors = [(1, 1, 1), (0, 0, 0.5)]  # White to deep blue
    cmap = LinearSegmentedColormap.from_list("custom_cmap", colors, N=100)
    
    # Plot the heatmap
    im = ax2.imshow(
        heatmap.T, 
        origin='lower', 
        aspect='auto',
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        cmap=cmap
    )
    
    plt.colorbar(im, ax=ax2, label='Number of Movies')
    
    # Add a diagonal line for perfect correlation
    ax2.plot([4, 10], [4, 10], 'r--', alpha=0.7, label='Perfect Agreement')
    
    # Add annotation for the counts in each cell
    for i in range(len(xedges)-1):
        for j in range(len(yedges)-1):
            if heatmap[i, j] > 0:
                ax2.text(
                    xedges[i] + (xedges[i+1] - xedges[i])/2,
                    yedges[j] + (yedges[j+1] - yedges[j])/2,
                    str(int(heatmap[i, j])),
                    ha='center',
                    va='center',
                    color='white' if heatmap[i, j] > heatmap.max()/2 else 'black',
                    fontsize=8
                )
    
    ax2.set_title('Heatmap: Rating Distribution', fontsize=14)
    ax2.set_xlabel('IMDb Rating', fontsize=12)
    ax2.set_ylabel('Planet B612 Rating', fontsize=12)
    ax2.grid(False)
    ax2.legend(loc='upper left', fontsize=8)
    
    # 3. Subcategory Analysis
    ax3 = plt.subplot(gs[1, 0])
    
    # Group by subcategory and calculate statistics
    subcategory_stats = df.groupby('subcategory').agg({
        'rating': ['count', 'mean'],
        'IMDb': ['mean'],
        'rating_difference': ['mean']
    })
    
    subcategory_stats.columns = ['count', 'b612_mean', 'imdb_mean', 'diff_mean']
    
    # Filter to only include subcategories with at least 3 movies
    subcategory_stats = subcategory_stats[subcategory_stats['count'] >= 3]
    subcategory_stats = subcategory_stats.sort_values('count', ascending=False)
    subcategory_stats = subcategory_stats.reset_index()
    
    # Plot average ratings by subcategory
    x = np.arange(len(subcategory_stats))
    width = 0.35
    
    ax3.bar(x - width/2, subcategory_stats['b612_mean'], width, label='Planet B612', color='blue', alpha=0.7)
    ax3.bar(x + width/2, subcategory_stats['imdb_mean'], width, label='IMDb', color='orange', alpha=0.7)
    
    # Add count labels
    for i, count in enumerate(subcategory_stats['count']):
        ax3.text(i, max(subcategory_stats['b612_mean'].max(), subcategory_stats['imdb_mean'].max()) + 0.2,
                f'n={count}', ha='center', fontsize=8)
    
    ax3.set_xticks(x)
    ax3.set_xticklabels(subcategory_stats['subcategory'], rotation=45, ha='right', fontsize=8)
    ax3.set_title('Average Ratings by Movie Subcategory', fontsize=14)
    ax3.set_ylabel('Average Rating', fontsize=12)
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=8)
    
    # 4. Timeline Analysis
    ax4 = plt.subplot(gs[1, 1])
    
    # Plot by release year
    release_year_stats = df.groupby('year_of_release').agg({
        'rating': ['count', 'mean'],
        'IMDb': ['mean'],
        'rating_difference': ['mean']
    })
    
    release_year_stats.columns = ['count', 'b612_mean', 'imdb_mean', 'diff_mean']
    release_year_stats = release_year_stats.reset_index()
    
    # Only include years with at least 2 movies and after 2000
    release_year_stats = release_year_stats[
        (release_year_stats['count'] >= 2) & 
        (release_year_stats['year_of_release'] >= 2000)
    ]
    
    # Plot mean ratings by release year
    ax4.plot(release_year_stats['year_of_release'], release_year_stats['b612_mean'], 
            marker='o', linestyle='-', linewidth=2, label='Planet B612', color='blue')
    ax4.plot(release_year_stats['year_of_release'], release_year_stats['imdb_mean'], 
            marker='s', linestyle='-', linewidth=2, label='IMDb', color='orange')
    
    # Add count labels
    for i, row in release_year_stats.iterrows():
        ax4.text(
            row['year_of_release'], 
            max(row['b612_mean'], row['imdb_mean']) + 0.1, 
            f'n={int(row["count"])}',
            ha='center',
            fontsize=8
        )
    
    # Add the difference as a bar plot at the bottom
    ax4_twin = ax4.twinx()
    ax4_twin.bar(
        release_year_stats['year_of_release'], 
        release_year_stats['diff_mean'], 
        alpha=0.2, 
        color=['red' if x < 0 else 'green' for x in release_year_stats['diff_mean']],
        width=0.8
    )
    ax4_twin.set_ylabel('Rating Difference', fontsize=10)
    ax4_twin.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    ax4.set_title('Average Ratings by Release Year (2000+)', fontsize=14)
    ax4.set_xlabel('Release Year', fontsize=12)
    ax4.set_ylabel('Average Rating', fontsize=12)
    ax4.legend(loc='upper left', fontsize=8)
    ax4.grid(True, alpha=0.3)
    
    # 5. Most Divergent Movies (Positive)
    ax5 = plt.subplot(gs[2, 0])
    
    most_positive = df.nlargest(7, 'rating_difference')
    
    ax5.barh(range(len(most_positive)), most_positive['rating_difference'], 
            color='green', alpha=0.7)
    
    ax5.set_yticks(range(len(most_positive)))
    ax5.set_yticklabels([f"{name[:20]}... ({b612:.1f} vs {imdb:.1f})" 
                        if len(name) > 20 else f"{name} ({b612:.1f} vs {imdb:.1f})" 
                        for name, b612, imdb in zip(
                            most_positive['name'],
                            most_positive['rating'],
                            most_positive['IMDb']
                        )], fontsize=8)
    
    for i, diff in enumerate(most_positive['rating_difference']):
        ax5.text(diff + 0.05, i, f'+{diff:.1f}', va='center', fontsize=8)
    
    ax5.set_title('Movies Where Planet B612 Rates Higher Than IMDb', fontsize=14)
    ax5.set_xlabel('Rating Difference (Planet B612 - IMDb)', fontsize=12)
    ax5.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    
    # 6. Most Divergent Movies (Negative)
    ax6 = plt.subplot(gs[2, 1])
    
    most_negative = df.nsmallest(7, 'rating_difference')
    
    ax6.barh(range(len(most_negative)), most_negative['rating_difference'], 
            color='red', alpha=0.7)
    
    ax6.set_yticks(range(len(most_negative)))
    ax6.set_yticklabels([f"{name[:20]}... ({b612:.1f} vs {imdb:.1f})" 
                        if len(name) > 20 else f"{name} ({b612:.1f} vs {imdb:.1f})" 
                        for name, b612, imdb in zip(
                            most_negative['name'],
                            most_negative['rating'],
                            most_negative['IMDb']
                        )], fontsize=8)
    
    for i, diff in enumerate(most_negative['rating_difference']):
        ax6.text(diff - 0.2, i, f'{diff:.1f}', va='center', fontsize=8)
    
    ax6.set_title('Movies Where Planet B612 Rates Lower Than IMDb', fontsize=14)
    ax6.set_xlabel('Rating Difference (Planet B612 - IMDb)', fontsize=12)
    ax6.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig('planet_b612_dashboard.png', dpi=300, bbox_inches='tight')
    plt.close()

def analyze_review_evolution(df):
    """
    Analyze how Planet B612's reviewing style has evolved over time
    """
    plt.figure(figsize=(14, 10))
    
    # Calculate yearly statistics of rating differences
    yearly_stats = df.groupby('year_of_review').agg({
        'rating_difference': ['mean', 'std', 'median', 'count']
    })
    
    yearly_stats.columns = ['mean_diff', 'std_diff', 'median_diff', 'count']
    yearly_stats = yearly_stats.reset_index()
    
    # Plot mean difference by year
    plt.errorbar(
        yearly_stats['year_of_review'], 
        yearly_stats['mean_diff'], 
        yerr=yearly_stats['std_diff'], 
        fmt='o-', 
        capsize=5, 
        linewidth=2,
        label='Mean Difference ± Std Dev'
    )
    
    # Add median line
    plt.plot(
        yearly_stats['year_of_review'], 
        yearly_stats['median_diff'], 
        's--', 
        color='red',
        label='Median Difference'
    )
    
    # Add zero line
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Add count annotations
    for i, row in yearly_stats.iterrows():
        plt.text(
            row['year_of_review'], 
            row['mean_diff'] + row['std_diff'] + 0.1, 
            f'n={int(row["count"])}',
            ha='center'
        )
    
    plt.title('Evolution of Rating Difference Over Time (Planet B612 - IMDb)', fontsize=16)
    plt.xlabel('Review Year', fontsize=14)
    plt.ylabel('Rating Difference', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('review_evolution.png', dpi=300, bbox_inches='tight')
    plt.close()

def analyze_imdb_popularity(df):
    """
    Analyze if there's any relationship between IMDb popularity (voters) and rating differences
    """
    # Only use entries with valid IMDb_voters data
    df_voters = df[df['IMDb_voters'].notna()].copy()
    
    # Log transform the number of voters for better visualization
    df_voters['log_voters'] = np.log10(df_voters['IMDb_voters'])
    
    plt.figure(figsize=(12, 8))
    
    # Create scatter plot
    scatter = plt.scatter(
        df_voters['log_voters'], 
        df_voters['rating_difference'],
        c=df_voters['IMDb'],
        cmap='viridis',
        alpha=0.7,
        s=80
    )
    
    # Add a horizontal line at 0 difference
    plt.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    
    # Add trend line
    z = np.polyfit(df_voters['log_voters'], df_voters['rating_difference'], 1)
    p = np.poly1d(z)
    plt.plot(
        np.sort(df_voters['log_voters']), 
        p(np.sort(df_voters['log_voters'])), 
        "r-", 
        alpha=0.7,
        label=f'Trend Line (y={z[0]:.2f}x+{z[1]:.2f})'
    )
    
    # Calculate correlation
    correlation, p_value = pearsonr(df_voters['log_voters'], df_voters['rating_difference'])
    
    # Add annotation for correlation
    plt.annotate(
        f'Correlation: {correlation:.2f}\np-value: {p_value:.4f}',
        xy=(0.05, 0.95),
        xycoords='axes fraction',
        fontsize=12,
        bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.7)
    )
    
    # Add colorbar for IMDb ratings
    cbar = plt.colorbar(scatter)
    cbar.set_label('IMDb Rating', fontsize=12)
    
    # Annotate a few notable points
    extreme_high = df_voters.nlargest(3, 'rating_difference')
    extreme_low = df_voters.nsmallest(3, 'rating_difference')
    popular = df_voters.nlargest(3, 'log_voters')
    
    for idx, row in pd.concat([extreme_high, extreme_low, popular]).drop_duplicates().iterrows():
        plt.annotate(
            row['name'][:15] + '...' if len(row['name']) > 15 else row['name'],
            xy=(row['log_voters'], row['rating_difference']),
            xytext=(5, 0),
            textcoords='offset points',
            fontsize=8,
            bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7)
        )
    
    # Format x-axis as actual voter numbers instead of log scale
    plt.xticks(
        np.arange(3, 7),
        ['1K', '10K', '100K', '1M']
    )
    
    plt.title('Relationship Between IMDb Popularity and Rating Difference', fontsize=16)
    plt.xlabel('Number of IMDb Voters (log scale)', fontsize=14)
    plt.ylabel('Rating Difference (Planet B612 - IMDb)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('popularity_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_critic_profile(df):
    """
    Create a radar chart to visualize Planet B612's critic "profile" compared to IMDb
    """
    # Get the top 6 subcategories by movie count
    top_subcategories = df['subcategory'].value_counts().nlargest(6).index.tolist()
    
    # Calculate mean ratings for these subcategories
    subcategory_data = df[df['subcategory'].isin(top_subcategories)].groupby('subcategory').agg({
        'rating': 'mean',
        'IMDb': 'mean'
    }).reindex(top_subcategories)
    
    # Number of variables
    N = len(subcategory_data)
    
    # Create angles for the radar chart
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # Close the loop
    
    # Add the data for both sources, and close the loop
    b612_values = subcategory_data['rating'].tolist()
    b612_values += b612_values[:1]
    
    imdb_values = subcategory_data['IMDb'].tolist()
    imdb_values += imdb_values[:1]
    
    # Create the radar chart
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    # Plot data
    ax.plot(angles, b612_values, 'o-', linewidth=2, label='Planet B612', color='blue')
    ax.fill(angles, b612_values, alpha=0.1, color='blue')
    
    ax.plot(angles, imdb_values, 'o-', linewidth=2, label='IMDb', color='orange')
    ax.fill(angles, imdb_values, alpha=0.1, color='orange')
    
    # Set labels and ticks
    labels = top_subcategories + [top_subcategories[0]]  # Close the loop
    ax.set_xticks(angles)
    ax.set_xticklabels(labels, fontsize=12)
    
    # Set y-limits (ratings)
    ax.set_ylim(5, 9)
    ax.set_yticks(np.arange(5, 9.1, 0.5))
    ax.set_yticklabels([f"{x:.1f}" for x in np.arange(5, 9.1, 0.5)], fontsize=10)
    
    # Add title and legend
    plt.title('Planet B612 vs IMDb: Genre Rating Profile', fontsize=16, y=1.1)
    plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))
    
    plt.tight_layout()
    plt.savefig('critic_profile.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """
    Main function to run all analyses
    """
    print("Starting Planet B612 Movie Critics Analysis...")
    
    # Load and clean data
    df = load_and_clean_data('Planet B612 Database .xlsx')
    print(f"Loaded {len(df)} movie reviews from Planet B612 Database.")
    
    # Calculate and print basic statistics
    calculate_basic_stats(df)
    
    # Create visualizations
    print("\nGenerating visualizations...")
    
    # Basic distributions and correlations
    plot_rating_distributions(df)
    plot_rating_correlation(df)
    create_heatmap_comparison(df)
    
    # Analysis by subcategory
    plot_subcategory_analysis(df)
    
    # Analysis over time
    plot_timeline_analysis(df)
    analyze_review_evolution(df)
    
    # Top and bottom movies
    plot_top_movies(df)
    plot_most_divergent(df)
    
    # Popularity analysis
    analyze_imdb_popularity(df)
    
    # Critic profile
    create_critic_profile(df)
    
    # Create comprehensive dashboard
    create_dashboard(df)
    
    print("\nAnalysis complete! All visualizations have been saved.")
    print("Files generated:")
    print("- rating_distributions.png")
    print("- rating_correlation.png")
    print("- rating_heatmap.png")
    print("- subcategory_analysis.png")
    print("- timeline_analysis.png")
    print("- review_evolution.png")
    print("- top_bottom_movies.png")
    print("- most_divergent.png")
    print("- popularity_analysis.png")
    print("- critic_profile.png")
    print("- planet_b612_dashboard.png (comprehensive summary)")

if __name__ == "__main__":
    main()