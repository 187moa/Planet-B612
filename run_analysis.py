#!/usr/bin/env python
"""
Planet B612 Movie Critic Analysis
=================================

This script runs a comprehensive analysis on the Planet B612 movie critic database,
generating visualizations and insights about their rating patterns compared to IMDb.

Run this script with:
    python run_analysis.py

Requirements:
    - pandas
    - numpy
    - matplotlib
    - seaborn
    - scipy
    - scikit-learn
    - wordcloud (optional for enhanced visualizations)

The script will generate various visualization files and reports.
"""

import os
import sys
import subprocess
import importlib.util

def check_dependencies():
    """Check if required Python packages are installed"""
    required_packages = [
        'pandas', 
        'numpy', 
        'matplotlib', 
        'seaborn', 
        'scipy', 
        'sklearn'
    ]
    
    optional_packages = [
        'wordcloud'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        if importlib.util.find_spec(package) is None:
            missing_packages.append(package)
    
    missing_optional = []
    for package in optional_packages:
        if importlib.util.find_spec(package) is None:
            missing_optional.append(package)
    
    if missing_packages:
        print(f"ERROR: The following required packages are missing: {', '.join(missing_packages)}")
        print("Please install them using pip:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    if missing_optional:
        print(f"NOTE: The following optional packages are missing: {', '.join(missing_optional)}")
        print("For enhanced visualizations, consider installing them:")
        print(f"pip install {' '.join(missing_optional)}")
    
    return True

def check_file_exists():
    """Check if the Excel file exists"""
    if not os.path.exists('Planet B612 Database .xlsx'):
        print("ERROR: 'Planet B612 Database .xlsx' file not found in the current directory.")
        return False
    return True

def run_visualization_script():
    """Run the visualization script"""
    print("\n" + "="*80)
    print("Running visualization script (planet_b612_analysis.py)...")
    print("="*80)
    
    try:
        import planet_b612_analysis
        planet_b612_analysis.main()
        return True
    except Exception as e:
        print(f"ERROR running visualization script: {str(e)}")
        return False

def run_insights_script():
    """Run the insights script"""
    print("\n" + "="*80)
    print("Running insights analysis script (planet_b612_insights.py)...")
    print("="*80)
    
    try:
        import planet_b612_insights
        planet_b612_insights.main()
        return True
    except Exception as e:
        print(f"ERROR running insights script: {str(e)}")
        return False

def main():
    """Main function to run the analysis"""
    print("Planet B612 Movie Critic Analysis Tool")
    print("="*80)
    
    # Check dependencies and file
    if not check_dependencies() or not check_file_exists():
        sys.exit(1)
    
    # Run the visualization script
    viz_success = run_visualization_script()
    
    # Run the insights script
    insights_success = run_insights_script()
    
    # Summary
    print("\n" + "="*80)
    print("Analysis Summary")
    print("="*80)
    
    if viz_success:
        print("✅ Visualizations generated successfully.")
        print("   The following files were created:")
        print("   - rating_distributions.png")
        print("   - rating_correlation.png")
        print("   - rating_heatmap.png")
        print("   - subcategory_analysis.png")
        print("   - timeline_analysis.png")
        print("   - review_evolution.png")
        print("   - top_bottom_movies.png")
        print("   - most_divergent.png")
        print("   - popularity_analysis.png")
        print("   - critic_profile.png")
        print("   - planet_b612_dashboard.png (comprehensive summary)")
    else:
        print("❌ Visualization generation failed.")
    
    if insights_success:
        print("✅ Insights report generated successfully.")
        print("   The following file was created:")
        print("   - planet_b612_analysis_report.md")
    else:
        print("❌ Insights report generation failed.")
    
    if viz_success and insights_success:
        print("\nAnalysis completed successfully! 🎉")
        print("You can now view the generated files in the current directory.")
    else:
        print("\nAnalysis completed with some errors. Please check the messages above.")

if __name__ == "__main__":
    main()