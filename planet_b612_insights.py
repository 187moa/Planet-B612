"""
Planet B612 Movie Critic Analysis - Insights and Interpretations

This script extends the visualizations with detailed analysis and insights into
Planet B612's unique critical voice and how it compares to IMDb consensus ratings.
It generates a report that highlights the distinctive aspects of Planet B612's critique style.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, ttest_ind
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.gridspec as gridspec
from wordcloud import WordCloud
import re
from collections import Counter

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
    df['subcategory'] = df['subcategory'].str.strip()
    
    # Convert years to integers where possible
    for col in ['year_of_release', 'year_of_review']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df

def generate_critic_identity_report(df):
    """
    Generate a detailed analysis of Planet B612's critic identity
    """
    # 1. Calculate the "dissent index" - how often Planet B612 disagrees with IMDb consensus
    df['significant_difference'] = abs(df['rating_difference']) > 1.0
    dissent_index = df['significant_difference'].mean() * 100
    
    # 2. Identify genres where Planet B612 is most distinctive
    genre_diff = df.groupby('subcategory').agg({
        'rating_difference': ['mean', 'std', 'count']
    })
    genre_diff.columns = ['mean_diff', 'std_diff', 'count']
    
    # Filter to genres with at least 3 movies
    distinctive_genres = genre_diff[genre_diff['count'] >= 3].sort_values('mean_diff')
    
    # 3. Calculate "harshness index" - how often Planet B612 rates lower than IMDb
    harshness_index = (df['rating_difference'] < 0).mean() * 100
    
    # 4. Calculate "enthusiasm index" - average rating for top rated movies
    enthusiasm_index = df.nlargest(10, 'rating')['rating'].mean()
    
    # 5. Calculate "alignment with critics" vs "alignment with audience"
    # (This is a hypothetical analysis; in a real scenario, you'd compare with critic scores)
    
    # 6. Analyze rating patterns for blockbusters vs indie films
    # Use IMDb voter count as a proxy for movie popularity
    df['blockbuster'] = df['IMDb_voters'] > 500000 if 'IMDb_voters' in df.columns else False
    blockbuster_diff = df.groupby('blockbuster')['rating_difference'].mean()
    
    # 7. Calculate temporal trends - is Planet B612 getting more or less aligned with IMDb over time?
    time_trend = df.groupby('year_of_review')['rating_difference'].mean()
    
    # 8. Calculate Planet B612's "confidence" - do they deviate more on certain types of films?
    confidence_by_category = df.groupby('category')['rating_difference'].agg(['mean', 'std'])
    
    # Generate the report
    report = [
        "# Planet B612 Critic Identity Analysis",
        "\n## Key Insights\n",
        f"- **Dissent Index**: {dissent_index:.1f}% of reviews differ significantly from IMDb consensus (>1.0 point difference)",
        f"- **Harshness Index**: Planet B612 rates movies lower than IMDb {harshness_index:.1f}% of the time",
        f"- **Enthusiasm Level**: Average rating for top 10 movies is {enthusiasm_index:.2f}/10",
    ]
    
    if 'blockbuster' in blockbuster_diff:
        report.append(f"- **Blockbuster Bias**: Planet B612 rates blockbusters {abs(blockbuster_diff[True]):.2f} points {'lower' if blockbuster_diff[True] < 0 else 'higher'} than IMDb on average")
    
    report.append("\n## Genre Specialization\n")
    
    # Add genre insights
    highest_genres = distinctive_genres.nlargest(3, 'mean_diff')
    lowest_genres = distinctive_genres.nsmallest(3, 'mean_diff')
    
    report.append("**Genres Where Planet B612 Is More Generous Than IMDb:**")
    for idx, row in highest_genres.iterrows():
        if row['mean_diff'] > 0:
            report.append(f"- {idx}: +{row['mean_diff']:.2f} points (based on {int(row['count'])} films)")
    
    report.append("\n**Genres Where Planet B612 Is More Critical Than IMDb:**")
    for idx, row in lowest_genres.iterrows():
        if row['mean_diff'] < 0:
            report.append(f"- {idx}: {row['mean_diff']:.2f} points (based on {int(row['count'])} films)")
    
    report.append("\n## Most Distinctive Reviews\n")
    
    # Add most distinctive reviews
    most_positive = df.nlargest(5, 'rating_difference')
    most_negative = df.nsmallest(5, 'rating_difference')
    
    report.append("**Films Planet B612 Appreciated More Than IMDb:**")
    for idx, row in most_positive.iterrows():
        report.append(f"- {row['name']}: {row['rating']:.1f} vs IMDb {row['IMDb']:.1f} (+{row['rating_difference']:.1f} points)")
    
    report.append("\n**Films Planet B612 Was More Critical Of Than IMDb:**")
    for idx, row in most_negative.iterrows():
        report.append(f"- {row['name']}: {row['rating']:.1f} vs IMDb {row['IMDb']:.1f} ({row['rating_difference']:.1f} points)")
    
    report.append("\n## Evolution Over Time\n")
    
    # Add time trend insights
    recent_years = time_trend[-3:]
    early_years = time_trend[:3]
    trend_direction = "more aligned with" if abs(recent_years.mean()) < abs(early_years.mean()) else "more divergent from"
    
    report.append(f"- Planet B612's ratings have become {trend_direction} IMDb consensus over time")
    report.append(f"- Average difference in early reviews ({early_years.index[0]}-{early_years.index[-1]}): {early_years.mean():.2f} points")
    report.append(f"- Average difference in recent reviews ({recent_years.index[0]}-{recent_years.index[-1]}): {recent_years.mean():.2f} points")
    
    report.append("\n## Critical Voice Analysis\n")
    
    # Calculate agreement with other critics
    agreement_with_consensus = 1 - (abs(df['rating_difference']).mean() / 10)
    
    # Add critical voice insights
    report.append(f"- **Agreement with Consensus**: {agreement_with_consensus*100:.1f}%")
    
    if 'comments' in df.columns:
        # Find most common descriptive words in comments
        words = ' '.join(df['comments'].dropna().astype(str)).lower()
        words = re.findall(r'\b[a-z]{4,15}\b', words)
        common_words = Counter(words).most_common(10)
        
        report.append("\n**Most Common Descriptive Terms in Reviews:**")
        for word, count in common_words:
            report.append(f"- {word}: {count} occurrences")
    
    return '\n'.join(report)

def find_signature_movies(df):
    """
    Identify films that best represent Planet B612's unique critical voice
    """
    # Calculate the "Planet B612 factor" - a compound score of how representative each movie is
    # of Planet B612's overall rating tendencies
    
    # 1. Z-score of how much the movie's rating differs from IMDb
    df['diff_zscore'] = (df['rating_difference'] - df['rating_difference'].mean()) / df['rating_difference'].std()
    
    # 2. Normalize the raw Planet B612 rating (how much they liked it in absolute terms)
    df['rating_norm'] = (df['rating'] - df['rating'].min()) / (df['rating'].max() - df['rating'].min())
    
    # 3. Weight for popularity (movies with more IMDb votes get higher weight)
    if 'IMDb_voters' in df.columns:
        df['log_voters'] = np.log10(df['IMDb_voters'])
        df['popularity_weight'] = (df['log_voters'] - df['log_voters'].min()) / (df['log_voters'].max() - df['log_voters'].min())
    else:
        df['popularity_weight'] = 0.5  # Default if we don't have voter data
    
    # 4. Calculate the compound score
    # - High when Planet B612 rates differently from consensus (high absolute z-score)
    # - High when the film is popular (high popularity weight)
    # - Different formulas for overpraised and underpraised films
    
    # For films Planet B612 rates higher than IMDb
    df_higher = df[df['rating_difference'] > 0].copy()
    if not df_higher.empty:
        df_higher['signature_score'] = (
            df_higher['diff_zscore'] * 0.5 + 
            df_higher['rating_norm'] * 0.3 + 
            df_higher['popularity_weight'] * 0.2
        )
    
    # For films Planet B612 rates lower than IMDb
    df_lower = df[df['rating_difference'] < 0].copy()
    if not df_lower.empty:
        df_lower['signature_score'] = (
            abs(df_lower['diff_zscore']) * 0.5 + 
            (1 - df_lower['rating_norm']) * 0.3 + 
            df_lower['popularity_weight'] * 0.2
        )
    
    # Get top signature films in each category
    top_higher = df_higher.nlargest(5, 'signature_score') if not df_higher.empty else pd.DataFrame()
    top_lower = df_lower.nlargest(5, 'signature_score') if not df_lower.empty else pd.DataFrame()
    
    # Generate the report
    report = [
        "# Planet B612's Signature Films",
        "\nThese films best represent Planet B612's unique critical voice compared to IMDb consensus.",
        "\n## Films Planet B612 Champions (Rated Significantly Higher Than IMDb)",
    ]
    
    if not top_higher.empty:
        for idx, row in top_higher.iterrows():
            report.append(f"- **{row['name']}** ({row['year_of_release']:.0f if pd.notna(row['year_of_release']) else 'N/A'}, {row['subcategory']})")
            report.append(f"  - Planet B612: {row['rating']:.1f}, IMDb: {row['IMDb']:.1f} (Difference: +{row['rating_difference']:.1f})")
            if 'IMDb_voters' in row and pd.notna(row['IMDb_voters']):
                report.append(f"  - IMDb Voters: {int(row['IMDb_voters']):,}")
            if 'comments' in row and pd.notna(row['comments']):
                report.append(f"  - Notes: {row['comments']}")
            report.append("")
    else:
        report.append("No films found where Planet B612 rates significantly higher than IMDb.")
    
    report.append("\n## Films Planet B612 Challenges (Rated Significantly Lower Than IMDb)")
    
    if not top_lower.empty:
        for idx, row in top_lower.iterrows():
            report.append(f"- **{row['name']}** ({row['year_of_release']:.0f if pd.notna(row['year_of_release']) else 'N/A'}, {row['subcategory']})")
            report.append(f"  - Planet B612: {row['rating']:.1f}, IMDb: {row['IMDb']:.1f} (Difference: {row['rating_difference']:.1f})")
            if 'IMDb_voters' in row and pd.notna(row['IMDb_voters']):
                report.append(f"  - IMDb Voters: {int(row['IMDb_voters']):,}")
            if 'comments' in row and pd.notna(row['comments']):
                report.append(f"  - Notes: {row['comments']}")
            report.append("")
    else:
        report.append("No films found where Planet B612 rates significantly lower than IMDb.")
    
    return '\n'.join(report)

def perform_cluster_analysis(df):
    """
    Perform cluster analysis to identify distinct types of reviews/ratings
    """
    # Select features for clustering
    features = ['rating', 'IMDb', 'rating_difference']
    if 'IMDb_voters' in df.columns:
        df['log_voters'] = np.log10(df['IMDb_voters'])
        features.append('log_voters')
    
    # Drop rows with missing values in these features
    df_cluster = df.dropna(subset=features).copy()
    
    # Standardize the features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df_cluster[features])
    
    # Determine optimal number of clusters (simplified approach)
    k = 3  # We'll use 3 clusters for simplicity
    
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    df_cluster['cluster'] = kmeans.fit_predict(scaled_features)
    
    # Analyze the clusters
    cluster_stats = df_cluster.groupby('cluster').agg({
        'rating': 'mean',
        'IMDb': 'mean',
        'rating_difference': 'mean',
    })
    
    if 'log_voters' in df_cluster.columns:
        cluster_stats['IMDb_voters'] = df_cluster.groupby('cluster')['IMDb_voters'].mean()
    
    # Add counts
    cluster_stats['count'] = df_cluster.groupby('cluster').size()
    
    # Add most common subcategories in each cluster
    cluster_subcats = {}
    for cluster in range(k):
        subcats = df_cluster[df_cluster['cluster'] == cluster]['subcategory'].value_counts().nlargest(3)
        cluster_subcats[cluster] = subcats
    
    # Add representative movies from each cluster
    cluster_examples = {}
    for cluster in range(k):
        # Find movies closest to the cluster centroid
        cluster_df = df_cluster[df_cluster['cluster'] == cluster]
        centroid = kmeans.cluster_centers_[cluster]
        
        # Calculate distance to centroid for each movie in the cluster
        distances = []
        for idx, row in cluster_df.iterrows():
            point = scaler.transform([row[features]])[0]
            distance = np.sqrt(np.sum((point - centroid) ** 2))
            distances.append((idx, distance))
        
        # Sort by distance and get the closest examples
        sorted_distances = sorted(distances, key=lambda x: x[1])
        closest_indices = [idx for idx, _ in sorted_distances[:3]]
        cluster_examples[cluster] = df.loc[closest_indices]
    
    # Generate cluster report
    report = [
        "# Cluster Analysis of Planet B612's Reviews",
        "\nWe identified distinct patterns in Planet B612's reviews by clustering them based on ratings, difference from IMDb, and popularity.",
    ]
    
    # Name the clusters based on their characteristics
    cluster_names = []
    for cluster in range(k):
        stats = cluster_stats.loc[cluster]
        
        if stats['rating_difference'] > 0.5:
            name = "Contrarian Praise"
        elif stats['rating_difference'] < -0.5:
            name = "Critical Stance"
        elif abs(stats['rating_difference']) <= 0.5 and stats['rating'] > 7.5:
            name = "Consensus Excellence"
        elif abs(stats['rating_difference']) <= 0.5 and stats['rating'] < 6:
            name = "Consensus Disappointment"
        else:
            name = "Moderate Alignment"
        
        cluster_names.append(name)
    
    # Add each cluster's details to the report
    for cluster in range(k):
        stats = cluster_stats.loc[cluster]
        name = cluster_names[cluster]
        
        report.append(f"\n## Cluster {cluster+1}: {name} ({stats['count']:.0f} films)")
        report.append(f"- **Average Planet B612 Rating**: {stats['rating']:.2f}/10")
        report.append(f"- **Average IMDb Rating**: {stats['IMDb']:.2f}/10")
        report.append(f"- **Average Difference**: {stats['rating_difference']:.2f} points")
        
        if 'IMDb_voters' in stats:
            report.append(f"- **Average IMDb Voters**: {stats['IMDb_voters']:.0f}")
        
        # Add common subcategories
        if cluster in cluster_subcats:
            report.append("\n**Most Common Genres:**")
            for subcat, count in cluster_subcats[cluster].items():
                report.append(f"- {subcat}: {count} films")
        
        # Add example films
        if cluster in cluster_examples:
            report.append("\n**Representative Films:**")
            for idx, row in cluster_examples[cluster].iterrows():
                report.append(f"- {row['name']} ({row['rating']:.1f} vs IMDb {row['IMDb']:.1f})")
    
    return '\n'.join(report)

def analyze_critical_philosophy(df):
    """
    Analyze Planet B612's critical philosophy and rating patterns
    """
    # Calculate how often Planet B612 aligns with, exceeds, or falls below IMDb
    align_threshold = 0.5  # Within 0.5 points is considered "aligned"
    
    aligned_count = sum(abs(df['rating_difference']) <= align_threshold)
    higher_count = sum(df['rating_difference'] > align_threshold)
    lower_count = sum(df['rating_difference'] < -align_threshold)
    
    align_percent = aligned_count / len(df) * 100
    higher_percent = higher_count / len(df) * 100
    lower_percent = lower_count / len(df) * 100
    
    # Calculate correlation between Planet B612 ratings and movie age
    if 'year_of_release' in df.columns:
        df['movie_age'] = 2025 - df['year_of_release']  # Using 2025 as current year
        age_corr, age_pval = pearsonr(df['movie_age'].dropna(), df['rating'].dropna())
    else:
        age_corr, age_pval = (0, 1)
    
    # Calculate if there's a bias against high-rated movies
    high_rated_imdb = df[df['IMDb'] >= 8.0]
    if not high_rated_imdb.empty:
        high_rated_diff = high_rated_imdb['rating_difference'].mean()
    else:
        high_rated_diff = 0
    
    # Calculate if there's a bias against low-rated movies
    low_rated_imdb = df[df['IMDb'] <= 6.0]
    if not low_rated_imdb.empty:
        low_rated_diff = low_rated_imdb['rating_difference'].mean()
    else:
        low_rated_diff = 0
    
    # Check if Planet B612 has a tendency to regress to the mean
    # (rating average vs popular movies more moderately)
    mean_rating = df['rating'].mean()
    high_correction = df[df['IMDb'] > mean_rating]['rating_difference'].mean()
    low_correction = df[df['IMDb'] < mean_rating]['rating_difference'].mean()
    
    # Generate the report
    report = [
        "# Analysis of Planet B612's Critical Philosophy",
        "\n## Rating Patterns\n",
        f"- **Alignment with IMDb**: {align_percent:.1f}% of ratings are within 0.5 points of IMDb",
        f"- **Higher than IMDb**: {higher_percent:.1f}% of ratings exceed IMDb by more than 0.5 points",
        f"- **Lower than IMDb**: {lower_percent:.1f}% of ratings fall below IMDb by more than 0.5 points",
    ]
    
    # Add insights about high vs low rated movies
    report.append("\n## Rating Tendencies\n")
    
    if high_rated_diff < -0.5:
        report.append(f"- **Critical of Acclaimed Films**: Planet B612 rates highly acclaimed films (IMDb ≥ 8.0) an average of {abs(high_rated_diff):.2f} points lower than IMDb")
    elif high_rated_diff > 0.5:
        report.append(f"- **Celebrates Acclaimed Films**: Planet B612 rates highly acclaimed films (IMDb ≥ 8.0) an average of {high_rated_diff:.2f} points higher than IMDb")
    else:
        report.append("- **Aligns on Acclaimed Films**: Planet B612 generally agrees with IMDb on highly rated films (IMDb ≥ 8.0)")
    
    if low_rated_diff < -0.5:
        report.append(f"- **Harsher on Poor Films**: Planet B612 rates poorly received films (IMDb ≤ 6.0) an average of {abs(low_rated_diff):.2f} points lower than IMDb")
    elif low_rated_diff > 0.5:
        report.append(f"- **Generous to Poor Films**: Planet B612 rates poorly received films (IMDb ≤ 6.0) an average of {low_rated_diff:.2f} points higher than IMDb")
    else:
        report.append("- **Aligns on Poor Films**: Planet B612 generally agrees with IMDb on poorly rated films (IMDb ≤ 6.0)")
    
    # Add insight about regression to the mean
    if high_correction < -0.5 and low_correction > 0.5:
        report.append("- **Moderating Influence**: Planet B612 tends to moderate extreme IMDb ratings, rating high IMDb films lower and low IMDb films higher")
    elif high_correction > 0.5 and low_correction < -0.5:
        report.append("- **Amplifying Influence**: Planet B612 tends to amplify IMDb's judgments, rating high IMDb films even higher and low IMDb films even lower")
    
    # Add insight about movie age correlation
    if abs(age_corr) > 0.2 and age_pval < 0.05:
        direction = "higher" if age_corr > 0 else "lower"
        report.append(f"- **Age Bias**: Planet B612 tends to rate older films {direction} than newer ones (correlation: {age_corr:.2f})")
    
    # Check if there's a notable difference in how Planet B612 rates movies vs TV shows
    if 'category' in df.columns:
        movies = df[df['category'].str.contains('Movie', case=False, na=False)]
        tv = df[df['category'].str.contains('TV', case=False, na=False)]
        
        if not movies.empty and not tv.empty:
            movie_diff = movies['rating_difference'].mean()
            tv_diff = tv['rating_difference'].mean()
            
            if abs(movie_diff - tv_diff) > 0.5:
                preferred = "TV shows" if tv_diff > movie_diff else "movies"
                report.append(f"- **Format Preference**: Planet B612 tends to be more generous toward {preferred} compared to {preferred.replace('shows', 'films') if preferred == 'TV shows' else 'TV shows'}")
    
    # Look at evolution of critic stance over time
    if 'year_of_review' in df.columns:
        years = sorted(df['year_of_review'].unique())
        if len(years) >= 3:
            first_year = df[df['year_of_review'] == years[0]]['rating_difference'].mean()
            last_year = df[df['year_of_review'] == years[-1]]['rating_difference'].mean()
            
            if abs(first_year - last_year) > 0.3:
                direction = "more generous" if last_year > first_year else "more critical"
                report.append(f"- **Evolution of Stance**: Planet B612 has become {direction} over time, with average rating difference changing from {first_year:.2f} in {years[0]:.0f} to {last_year:.2f} in {years[-1]:.0f}")
    
    # Add conclusion about Planet B612's critical voice
    report.append("\n## Critical Voice\n")
    
    if lower_percent > higher_percent + 20:
        report.append("- **Critic Style**: Planet B612 demonstrates a primarily critical voice, often challenging popular consensus by providing more stringent evaluations")
    elif higher_percent > lower_percent + 20:
        report.append("- **Critic Style**: Planet B612 demonstrates a primarily appreciative voice, often championing films beyond their popular reception")
    elif align_percent > 60:
        report.append("- **Critic Style**: Planet B612 demonstrates a primarily consensual voice, generally aligning with broader audience sentiment while providing nuanced context")
    else:
        report.append("- **Critic Style**: Planet B612 demonstrates a balanced critical voice, alternating between praise and criticism based on the specific merits of each work")
    
    return '\n'.join(report)

def identify_blind_spots(df):
    """
    Identify potential blind spots or biases in Planet B612's reviewing patterns
    """
    # Look for patterns in the data that might indicate blind spots or biases
    
    # 1. Check if certain subcategories/genres are underrepresented
    genre_counts = df['subcategory'].value_counts()
    total_reviews = len(df)
    
    # 2. Check if ratings show significant gender bias (this would require director/lead gender data)
    # For now, we'll focus on other patterns
    
    # 3. Check if certain years or eras are underrepresented
    if 'year_of_release' in df.columns:
        era_counts = pd.cut(df['year_of_release'], bins=[1900, 1950, 1970, 1990, 2000, 2010, 2020, 2030]).value_counts().sort_index()
    
    # 4. Check if Planet B612 shows bias toward certain countries/regions
    # (This would require country of origin data)
    
    # 5. Check if there's bias for/against big budget films
    # (Use IMDb voters as a proxy for film popularity/budget)
    
    # Generate the report
    report = [
        "# Potential Blind Spots in Planet B612's Coverage",
        "\nThis analysis identifies potential gaps or biases in Planet B612's review patterns.",
        "\n## Genre Coverage\n"
    ]
    
    # Add genre coverage insights
    dominant_genres = genre_counts.nlargest(3)
    ignored_genres = []
    
    for genre, count in dominant_genres.items():
        percentage = count / total_reviews * 100
        report.append(f"- **{genre}**: {count} reviews ({percentage:.1f}% of total coverage)")
    
    # List of common film genres that might be missing or underrepresented
    common_genres = ["Drama", "Comedy", "Action", "Horror", "Sci-fi", "Thriller", 
                    "Romance", "Animation", "Documentary", "Fantasy", "Adventure"]
    
    for genre in common_genres:
        if genre not in genre_counts or genre_counts[genre] < 3:
            ignored_genres.append(genre)
    
    if ignored_genres:
        report.append("\n**Potentially Underrepresented Genres:**")
        for genre in ignored_genres:
            report.append(f"- {genre}")
    
    # Add era coverage insights
    if 'year_of_release' in df.columns:
        report.append("\n## Era Coverage\n")
        
        for era_range, count in era_counts.items():
            percentage = count / total_reviews * 100
            era_str = str(era_range).replace("(", "").replace("]", "").replace(", ", "-")
            report.append(f"- **{era_str}**: {count} reviews ({percentage:.1f}% of total coverage)")
        
        # Check for significant gaps
        if era_counts.min() < 3:
            underrepresented_eras = [str(era).replace("(", "").replace("]", "").replace(", ", "-") 
                                   for era, count in era_counts.items() if count < 3]
            report.append("\n**Potentially Underrepresented Eras:**")
            for era in underrepresented_eras:
                report.append(f"- {era}")
    
    # Add rating distribution insights
    report.append("\n## Rating Distribution\n")
    
    # Check if ratings are well-distributed or clustered
    rating_counts = df['rating'].value_counts().sort_index()
    rating_distribution = rating_counts / rating_counts.sum() * 100
    
    # Check if any ranges are overrepresented
    low_ratings = rating_distribution[rating_distribution.index < 5].sum()
    mid_ratings = rating_distribution[(rating_distribution.index >= 5) & (rating_distribution.index < 8)].sum()
    high_ratings = rating_distribution[rating_distribution.index >= 8].sum()
    
    report.append(f"- **Low Ratings (0-4.9)**: {low_ratings:.1f}% of reviews")
    report.append(f"- **Mid Ratings (5-7.9)**: {mid_ratings:.1f}% of reviews")
    report.append(f"- **High Ratings (8-10)**: {high_ratings:.1f}% of reviews")
    
    if high_ratings > 50:
        report.append("\n⚠️ **Potential Positive Bias**: Over half of all ratings are 8 or above, possibly indicating a bias toward positive reviews")
    elif low_ratings > 30:
        report.append("\n⚠️ **Potential Negative Bias**: A large portion of ratings are below 5, possibly indicating a bias toward negative reviews")
    
    # Check if there's a bias toward specific ratings (e.g., avoiding certain numbers)
    common_ratings = rating_counts.nlargest(3)
    rare_ratings = rating_counts[rating_counts < 2]
    
    report.append("\n**Most Common Rating Values:**")
    for rating, count in common_ratings.items():
        percentage = count / total_reviews * 100
        report.append(f"- {rating}: {count} reviews ({percentage:.1f}%)")
    
    if not rare_ratings.empty:
        report.append("\n**Rarely Used Rating Values:**")
        for rating, count in rare_ratings.items():
            report.append(f"- {rating}: {count} reviews")
    
    return '\n'.join(report)

def generate_comprehensive_report(df):
    """
    Generate a comprehensive report with all analyses
    """
    critic_identity = generate_critic_identity_report(df)
    signature_films = find_signature_movies(df)
    cluster_analysis = perform_cluster_analysis(df)
    critical_philosophy = analyze_critical_philosophy(df)
    blind_spots = identify_blind_spots(df)
    
    report = [
        "# Planet B612 Movie Critic: Comprehensive Analysis",
        "\nThis report provides a detailed analysis of Planet B612's critical voice, rating patterns, and distinctive characteristics.",
        "\n---\n",
        critic_identity,
        "\n---\n",
        signature_films,
        "\n---\n",
        cluster_analysis,
        "\n---\n",
        critical_philosophy,
        "\n---\n",
        blind_spots
    ]
    
    return '\n'.join(report)

def main():
    """
    Main function to generate the report
    """
    print("Starting Planet B612 Movie Critics In-Depth Analysis...")
    
    # Load and clean data
    df = load_and_clean_data('Planet B612 Database .xlsx')
    print(f"Loaded {len(df)} movie reviews from Planet B612 Database.")
    
    # Generate comprehensive report
    print("\nGenerating in-depth analysis report...")
    report = generate_comprehensive_report(df)
    
    # Save report to a file
    with open('planet_b612_analysis_report.md', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print("\nAnalysis complete! The report has been saved to 'planet_b612_analysis_report.md'")

if __name__ == "__main__":
    main()