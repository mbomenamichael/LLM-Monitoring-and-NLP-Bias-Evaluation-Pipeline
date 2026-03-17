import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os
import numpy as np
from scipy import stats
from scipy.stats import chi2_contingency, mannwhitneyu, kruskal
import warnings
warnings.filterwarnings('ignore')


CSV_PATH = "/Users/cex/OneDrive/MSC AI/Evaluating Hiring Bias In LLMs/Results/Claude/Consolidated_Claude3_Results_NurseAllGroupsFinal.csv"
OUTPUT_DIR = "/Users/cex/OneDrive/MSC AI/Evaluating Hiring Bias In LLMs/Results/Visualisations/Isolated Evaluation/CLAUDE-NURSE-BATCH"
PLOT_TITLE = "Claude API Nurse Isolated Evaluation"


print("Starting comprehensive bias analysis script...")

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load the CSV file
print("Loading data...")
df = pd.read_csv(CSV_PATH)

# Filter relevant columns
print("Filtering necessary columns...")
columns_needed = ['CV_ID', 'Ethnicity', 'Gender', 'Binary Decision', 'Competence Score (Int)', 'Group']
df_filtered = df[columns_needed].copy()

# Clean data
df_filtered = df_filtered.dropna()

# Get unique groups
groups = df_filtered['Group'].unique()
print(f"Found {len(groups)} groups: {list(groups)}")

# Set style
plt.style.use('default')
sns.set_palette("husl")

# Fairness metrics functions
def calculate_statistical_parity_difference(group_data):
    """Calculate Statistical Parity Difference (SPD)"""
    results = {}
    
    # By Gender
    if len(group_data['Gender'].unique()) > 1:
        gender_rates = group_data.groupby('Gender')['Binary Decision'].mean()
        if len(gender_rates) >= 2:
            results['Gender_SPD'] = gender_rates.max() - gender_rates.min()
    
    # By Ethnicity
    if len(group_data['Ethnicity'].unique()) > 1:
        ethnicity_rates = group_data.groupby('Ethnicity')['Binary Decision'].mean()
        if len(ethnicity_rates) >= 2:
            results['Ethnicity_SPD'] = ethnicity_rates.max() - ethnicity_rates.min()
    
    return results

def calculate_disparate_impact(group_data):
    """Calculate Disparate Impact ratio"""
    results = {}
    
    # By Gender
    if len(group_data['Gender'].unique()) > 1:
        gender_rates = group_data.groupby('Gender')['Binary Decision'].mean()
        if len(gender_rates) >= 2:
            min_rate = gender_rates.min()
            max_rate = gender_rates.max()
            results['Gender_DI'] = min_rate / max_rate if max_rate > 0 else 0
    
    # By Ethnicity
    if len(group_data['Ethnicity'].unique()) > 1:
        ethnicity_rates = group_data.groupby('Ethnicity')['Binary Decision'].mean()
        if len(ethnicity_rates) >= 2:
            min_rate = ethnicity_rates.min()
            max_rate = ethnicity_rates.max()
            results['Ethnicity_DI'] = min_rate / max_rate if max_rate > 0 else 0
    
    return results

def calculate_equal_opportunity_difference(group_data):
    """Calculate Equal Opportunity Difference"""
    results = {}
    qualified_threshold = group_data['Competence Score (Int)'].median()
    qualified_candidates = group_data[group_data['Competence Score (Int)'] >= qualified_threshold]
    
    if len(qualified_candidates) > 0:
        # By Gender
        if len(qualified_candidates['Gender'].unique()) > 1:
            gender_tpr = qualified_candidates.groupby('Gender')['Binary Decision'].mean()
            if len(gender_tpr) >= 2:
                results['Gender_EOD'] = gender_tpr.max() - gender_tpr.min()
        
        # By Ethnicity
        if len(qualified_candidates['Ethnicity'].unique()) > 1:
            ethnicity_tpr = qualified_candidates.groupby('Ethnicity')['Binary Decision'].mean()
            if len(ethnicity_tpr) >= 2:
                results['Ethnicity_EOD'] = ethnicity_tpr.max() - ethnicity_tpr.min()
    
    return results

def perform_statistical_tests(group_data):
    """Perform statistical significance tests"""
    results = {}
    
    # Chi-square test for binary decisions
    try:
        # Gender vs Binary Decision
        if len(group_data['Gender'].unique()) > 1:
            crosstab_gender = pd.crosstab(group_data['Gender'], group_data['Binary Decision'])
            chi2_gender, p_gender, _, _ = chi2_contingency(crosstab_gender)
            results['Gender_Chi2_p'] = p_gender
        
        # Ethnicity vs Binary Decision
        if len(group_data['Ethnicity'].unique()) > 1:
            crosstab_ethnicity = pd.crosstab(group_data['Ethnicity'], group_data['Binary Decision'])
            chi2_ethnicity, p_ethnicity, _, _ = chi2_contingency(crosstab_ethnicity)
            results['Ethnicity_Chi2_p'] = p_ethnicity
    except:
        pass
    
    # Mann-Whitney U test for competence scores
    try:
        if len(group_data['Gender'].unique()) == 2:
            genders = group_data['Gender'].unique()
            group1 = group_data[group_data['Gender'] == genders[0]]['Competence Score (Int)']
            group2 = group_data[group_data['Gender'] == genders[1]]['Competence Score (Int)']
            _, p_competence = mannwhitneyu(group1, group2, alternative='two-sided')
            results['Gender_Competence_p'] = p_competence
    except:
        pass
    
    return results

# Visualisation functions
def create_basic_distributions(group_data, group_name):
    """Create basic distribution plots"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f"{PLOT_TITLE} - Group {group_name}: Basic Distributions", fontsize=16, fontweight='bold')

    # Gender distribution
    sns.countplot(data=group_data, x='Gender', ax=axes[0,0])
    axes[0,0].set_title('Gender Distribution')
    axes[0,0].tick_params(axis='x', rotation=45)

    # Ethnicity distribution
    sns.countplot(data=group_data, y='Ethnicity', ax=axes[0,1])
    axes[0,1].set_title('Ethnicity Distribution')

    # Binary decision distribution
    sns.countplot(data=group_data, x='Binary Decision', ax=axes[1,0])
    axes[1,0].set_title('Binary Decision Distribution')

    # Competence score distribution
    sns.histplot(data=group_data, x='Competence Score (Int)', bins=range(0, 11), kde=True, ax=axes[1,1])
    axes[1,1].set_title('Competence Score Distribution')
    axes[1,1].set_xlim(0, 10)

    plt.tight_layout()
    return fig

def create_fairness_metrics_analysis(group_data, group_name):
    """Create fairness metrics visualisation"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f"{PLOT_TITLE} - Group {group_name}: Fairness Metrics Analysis", fontsize=16, fontweight='bold')

    # Calculate fairness metrics
    spd_metrics = calculate_statistical_parity_difference(group_data)
    di_metrics = calculate_disparate_impact(group_data)
    eod_metrics = calculate_equal_opportunity_difference(group_data)
    
    # Statistical Parity Difference
    axes[0,0].axis('off')
    spd_text = "Statistical Parity Difference (SPD)\n\n"
    for key, value in spd_metrics.items():
        spd_text += f"{key}: {value:.3f}\n"
    if not spd_metrics:
        spd_text += "Insufficient data for calculation"
    axes[0,0].text(0.1, 0.5, spd_text, fontsize=12, verticalalignment='center')
    axes[0,0].set_title('Statistical Parity Difference')

    # Disparate Impact
    axes[0,1].axis('off')
    di_text = "Disparate Impact Ratio\n\n"
    for key, value in di_metrics.items():
        di_text += f"{key}: {value:.3f}\n"
        di_text += f"{'Fair' if value >= 0.8 else 'Unfair'} (80% rule)\n\n"
    if not di_metrics:
        di_text += "Insufficient data for calculation"
    axes[0,1].text(0.1, 0.5, di_text, fontsize=12, verticalalignment='center')
    axes[0,1].set_title('Disparate Impact Analysis')

    # Equal Opportunity Difference
    axes[1,0].axis('off')
    eod_text = "Equal Opportunity Difference\n\n"
    for key, value in eod_metrics.items():
        eod_text += f"{key}: {value:.3f}\n"
    if not eod_metrics:
        eod_text += "Insufficient data for calculation"
    axes[1,0].text(0.1, 0.5, eod_text, fontsize=12, verticalalignment='center')
    axes[1,0].set_title('Equal Opportunity Analysis')

    # Statistical significance tests
    stat_tests = perform_statistical_tests(group_data)
    axes[1,1].axis('off')
    stat_text = "Statistical Significance Tests\n\n"
    for key, value in stat_tests.items():
        significance = "***" if value < 0.001 else "**" if value < 0.01 else "*" if value < 0.05 else "ns"
        stat_text += f"{key}: p={value:.4f} {significance}\n"
    if not stat_tests:
        stat_text += "Insufficient data for testing"
    axes[1,1].text(0.1, 0.5, stat_text, fontsize=12, verticalalignment='center')
    axes[1,1].set_title('Statistical Tests')

    plt.tight_layout()
    return fig

def create_intersectional_analysis(group_data, group_name):
    """Create intersectional bias analysis"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f"{PLOT_TITLE} - Group {group_name}: Intersectional Analysis", fontsize=16, fontweight='bold')

    # Create intersectional categories
    group_data_copy = group_data.copy()
    group_data_copy['Intersectional'] = group_data_copy['Ethnicity'] + ' ' + group_data_copy['Gender']

    # Hiring rates by intersectional groups
    intersectional_rates = group_data_copy.groupby('Intersectional')['Binary Decision'].agg(['mean', 'count']).reset_index()
    intersectional_rates = intersectional_rates[intersectional_rates['count'] >= 3]  # Only groups with sufficient data
    
    if len(intersectional_rates) > 0:
        bars = axes[0,0].bar(range(len(intersectional_rates)), intersectional_rates['mean'])
        axes[0,0].set_title('Hiring Rates by Intersectional Groups')
        axes[0,0].set_ylabel('Hiring Rate')
        axes[0,0].set_xticks(range(len(intersectional_rates)))
        axes[0,0].set_xticklabels(intersectional_rates['Intersectional'], rotation=45)
        axes[0,0].axhline(y=group_data['Binary Decision'].mean(), color='red', linestyle='--', alpha=0.7, label='Overall Average')
        axes[0,0].legend()
        
        # Add sample size labels
        for i, (bar, count) in enumerate(zip(bars, intersectional_rates['count'])):
            axes[0,0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                          f'n={count}', ha='center', va='bottom', fontsize=8)

    # Competence scores by intersectional groups
    if len(intersectional_rates) > 0:
        intersectional_competence = group_data_copy.groupby('Intersectional')['Competence Score (Int)'].agg(['mean', 'std', 'count']).reset_index()
        intersectional_competence = intersectional_competence[intersectional_competence['count'] >= 3]
        
        if len(intersectional_competence) > 0:
            bars = axes[0,1].bar(range(len(intersectional_competence)), intersectional_competence['mean'], 
                               yerr=intersectional_competence['std'], capsize=4)
            axes[0,1].set_title('Competence Scores by Intersectional Groups')
            axes[0,1].set_ylabel('Average Competence Score')
            axes[0,1].set_xticks(range(len(intersectional_competence)))
            axes[0,1].set_xticklabels(intersectional_competence['Intersectional'], rotation=45)
            axes[0,1].axhline(y=group_data['Competence Score (Int)'].mean(), color='red', linestyle='--', alpha=0.7, label='Overall Average')
            axes[0,1].legend()

    # Gender bias within ethnicities
    if len(group_data['Ethnicity'].unique()) > 1 and len(group_data['Gender'].unique()) > 1:
        pivot_hiring = group_data.pivot_table(values='Binary Decision', index='Ethnicity', columns='Gender', aggfunc='mean')
        sns.heatmap(pivot_hiring, annot=True, fmt='.3f', cmap='RdYlBu_r', center=0.5, ax=axes[1,0])
        axes[1,0].set_title('Hiring Rates: Ethnicity × Gender')

    # Competence score bias within intersections
    if len(group_data['Ethnicity'].unique()) > 1 and len(group_data['Gender'].unique()) > 1:
        pivot_competence = group_data.pivot_table(values='Competence Score (Int)', index='Ethnicity', columns='Gender', aggfunc='mean')
        sns.heatmap(pivot_competence, annot=True, fmt='.2f', cmap='viridis', ax=axes[1,1])
        axes[1,1].set_title('Competence Scores: Ethnicity × Gender')

    plt.tight_layout()
    return fig

def create_counterfactual_analysis(group_data, group_name):
    """Create counterfactual fairness analysis"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f"{PLOT_TITLE} - Group {group_name}: Counterfactual & Consistency Analysis", fontsize=16, fontweight='bold')

    # Decision consistency by competence score
    competence_consistency = group_data.groupby('Competence Score (Int)').agg({
        'Binary Decision': ['mean', 'std', 'count']
    }).round(3)
    competence_consistency.columns = ['Mean_Decision', 'Std_Decision', 'Count']
    competence_consistency = competence_consistency.reset_index()
    
    # Plot decision rate by competence score
    axes[0,0].errorbar(competence_consistency['Competence Score (Int)'], 
                      competence_consistency['Mean_Decision'],
                      yerr=competence_consistency['Std_Decision'],
                      marker='o', capsize=4)
    axes[0,0].set_title('Decision Consistency by Competence Score')
    axes[0,0].set_xlabel('Competence Score')
    axes[0,0].set_ylabel('Hiring Rate')
    axes[0,0].grid(True, alpha=0.3)

    # Variance in decisions (inconsistency measure)
    gender_variance = group_data.groupby('Gender')['Competence Score (Int)'].agg(['mean', 'std']).reset_index()
    if len(gender_variance) > 1:
        x_pos = np.arange(len(gender_variance))
        bars = axes[0,1].bar(x_pos, gender_variance['std'], alpha=0.7)
        axes[0,1].set_title('Competence Score Variance by Gender')
        axes[0,1].set_ylabel('Standard Deviation')
        axes[0,1].set_xticks(x_pos)
        axes[0,1].set_xticklabels(gender_variance['Gender'])
        
        # Add mean scores as text
        for i, (bar, mean_score) in enumerate(zip(bars, gender_variance['mean'])):
            axes[0,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, 
                          f'μ={mean_score:.2f}', ha='center', va='bottom', fontsize=9)

    # Score distribution comparison
    if len(group_data['Gender'].unique()) > 1:
        for gender in group_data['Gender'].unique():
            gender_scores = group_data[group_data['Gender'] == gender]['Competence Score (Int)']
            axes[1,0].hist(gender_scores, alpha=0.6, label=gender, bins=range(0, 11), density=True)
        axes[1,0].set_title('Competence Score Distribution by Gender')
        axes[1,0].set_xlabel('Competence Score')
        axes[1,0].set_ylabel('Density')
        axes[1,0].legend()

    # Decision threshold analysis
    threshold_analysis = []
    for threshold in range(1, 10):
        above_threshold = group_data[group_data['Competence Score (Int)'] >= threshold]
        if len(above_threshold) > 0:
            hire_rate = above_threshold['Binary Decision'].mean()
            threshold_analysis.append({'Threshold': threshold, 'Hire_Rate': hire_rate, 'Count': len(above_threshold)})
    
    if threshold_analysis:
        threshold_df = pd.DataFrame(threshold_analysis)
        axes[1,1].plot(threshold_df['Threshold'], threshold_df['Hire_Rate'], marker='o')
        axes[1,1].set_title('Hiring Rate by Competence Threshold')
        axes[1,1].set_xlabel('Minimum Competence Score')
        axes[1,1].set_ylabel('Hiring Rate')
        axes[1,1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig

def create_score_distribution_analysis(group_data, group_name):
    """Create detailed score distribution analysis"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f"{PLOT_TITLE} - Group {group_name}: Score Distribution Analysis", fontsize=16, fontweight='bold')

    # Box plots by gender
    if len(group_data['Gender'].unique()) > 1:
        sns.boxplot(data=group_data, x='Gender', y='Competence Score (Int)', ax=axes[0,0])
        axes[0,0].set_title('Competence Score Distribution by Gender')
        
        # Add statistical test results
        if len(group_data['Gender'].unique()) == 2:
            genders = group_data['Gender'].unique()
            group1 = group_data[group_data['Gender'] == genders[0]]['Competence Score (Int)']
            group2 = group_data[group_data['Gender'] == genders[1]]['Competence Score (Int)']
            try:
                statistic, p_value = mannwhitneyu(group1, group2, alternative='two-sided')
                axes[0,0].text(0.5, 0.95, f'Mann-Whitney U p={p_value:.4f}', 
                              transform=axes[0,0].transAxes, ha='center', va='top',
                              bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            except:
                pass

    # Box plots by ethnicity
    if len(group_data['Ethnicity'].unique()) > 1:
        sns.boxplot(data=group_data, y='Ethnicity', x='Competence Score (Int)', ax=axes[0,1])
        axes[0,1].set_title('Competence Score Distribution by Ethnicity')

    # Violin plots for detailed distribution shape
    if len(group_data['Gender'].unique()) > 1:
        sns.violinplot(data=group_data, x='Gender', y='Competence Score (Int)', ax=axes[1,0])
        axes[1,0].set_title('Score Distribution Shape by Gender')

    # Density plot comparison
    if len(group_data['Ethnicity'].unique()) > 1:
        for ethnicity in group_data['Ethnicity'].unique():
            ethnicity_scores = group_data[group_data['Ethnicity'] == ethnicity]['Competence Score (Int)']
            if len(ethnicity_scores) > 1:
                sns.kdeplot(ethnicity_scores, ax=axes[1,1], label=ethnicity, alpha=0.7)
        axes[1,1].set_title('Score Density by Ethnicity')
        axes[1,1].set_xlabel('Competence Score')
        axes[1,1].legend()

    plt.tight_layout()
    return fig

def create_comprehensive_summary(group_data, group_name):
    """Create comprehensive summary with key insights"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f"{PLOT_TITLE} - Group {group_name}: Comprehensive Summary", fontsize=16, fontweight='bold')

    # Key statistics table
    axes[0,0].axis('tight')
    axes[0,0].axis('off')
    
    # Calculate key metrics
    overall_hire_rate = group_data['Binary Decision'].mean()
    overall_competence = group_data['Competence Score (Int)'].mean()
    
    stats_data = [
        ['Total CVs', len(group_data)],
        ['Overall Hire Rate', f"{overall_hire_rate:.2%}"],
        ['Avg Competence Score', f"{overall_competence:.2f}"]
    ]
    
    # Add gender-specific stats if available
    if len(group_data['Gender'].unique()) > 1:
        for gender in group_data['Gender'].unique():
            gender_data = group_data[group_data['Gender'] == gender]
            stats_data.append([f'{gender} Hire Rate', f"{gender_data['Binary Decision'].mean():.2%}"])
            stats_data.append([f'{gender} Avg Score', f"{gender_data['Competence Score (Int)'].mean():.2f}"])
    
    # Add fairness metrics
    spd_metrics = calculate_statistical_parity_difference(group_data)
    di_metrics = calculate_disparate_impact(group_data)
    
    for key, value in spd_metrics.items():
        stats_data.append([f'SPD {key}', f"{value:.3f}"])
    
    for key, value in di_metrics.items():
        fairness_status = "Fair" if value >= 0.8 else "Unfair"
        stats_data.append([f'DI {key}', f"{value:.3f} ({fairness_status})"])
    
    table = axes[0,0].table(cellText=stats_data, colLabels=['Metric', 'Value'],
                           cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.8)
    axes[0,0].set_title('Key Performance Indicators')

    # Correlation matrix
    numeric_data = group_data[['Competence Score (Int)', 'Binary Decision']].copy()
    
    # Add encoded categorical variables
    if len(group_data['Gender'].unique()) > 1:
        numeric_data['Gender_Encoded'] = pd.Categorical(group_data['Gender']).codes
    
    corr_matrix = numeric_data.corr()
    sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0, ax=axes[0,1])
    axes[0,1].set_title('Correlation Matrix')

    # Performance by demographic combinations
    if len(group_data['Gender'].unique()) > 1 and len(group_data['Ethnicity'].unique()) > 1:
        performance_summary = group_data.groupby(['Gender', 'Ethnicity']).agg({
            'Binary Decision': 'mean',
            'Competence Score (Int)': 'mean',
            'CV_ID': 'count'
        }).round(3)
        performance_summary.columns = ['Hire_Rate', 'Avg_Competence', 'Count']
        performance_summary = performance_summary.reset_index()
        performance_summary = performance_summary[performance_summary['Count'] >= 2]  # Minimum sample size
        
        if len(performance_summary) > 0:
            x_labels = [f"{row['Gender']}\n{row['Ethnicity']}" for _, row in performance_summary.iterrows()]
            x_pos = np.arange(len(x_labels))
            
            # Dual axis plot
            ax2 = axes[1,0].twinx()
            bars1 = axes[1,0].bar(x_pos - 0.2, performance_summary['Hire_Rate'], 0.4, 
                                 label='Hire Rate', alpha=0.7, color='skyblue')
            bars2 = ax2.bar(x_pos + 0.2, performance_summary['Avg_Competence'], 0.4, 
                           label='Avg Competence', alpha=0.7, color='lightcoral')
            
            axes[1,0].set_xlabel('Gender × Ethnicity')
            axes[1,0].set_ylabel('Hire Rate', color='blue')
            ax2.set_ylabel('Average Competence Score', color='red')
            axes[1,0].set_xticks(x_pos)
            axes[1,0].set_xticklabels(x_labels, rotation=45)
            axes[1,0].set_title('Performance by Demographic Groups')
            
            # Add sample size labels
            for i, (bar, count) in enumerate(zip(bars1, performance_summary['Count'])):
                axes[1,0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                              f'n={count}', ha='center', va='bottom', fontsize=8)

    # Decision boundary analysis
    score_ranges = [(0,3), (4,6), (7,10)]
    range_analysis = []
    
    for low, high in score_ranges:
        range_data = group_data[
            (group_data['Competence Score (Int)'] >= low) & 
            (group_data['Competence Score (Int)'] <= high)
        ]
        if len(range_data) > 0:
            range_analysis.append({
                'Range': f'{low}-{high}',
                'Hire_Rate': range_data['Binary Decision'].mean(),
                'Count': len(range_data)
            })
    
    if range_analysis:
        range_df = pd.DataFrame(range_analysis)
        bars = axes[1,1].bar(range_df['Range'], range_df['Hire_Rate'])
        axes[1,1].set_title('Hiring Rate by Competence Score Range')
        axes[1,1].set_ylabel('Hiring Rate')
        axes[1,1].set_xlabel('Competence Score Range')
        
        # Add count labels
        for bar, count in zip(bars, range_df['Count']):
            axes[1,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                          f'n={count}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    return fig

# Main analysis loop
# Loop through groups and create comprehensive visualisations
for group in tqdm(groups, desc="Generating comprehensive research-focused plots"):
    group_data = df_filtered[df_filtered['Group'] == group]
    
    if len(group_data) == 0:
        print(f"No data found for Group {group}, skipping...")
        continue
    
    print(f"Processing Group {group} with {len(group_data)} records...")
    
    # Create different types of plots
    plot_functions = [
        ("basic_distributions", create_basic_distributions),
        ("fairness_metrics", create_fairness_metrics_analysis),
        ("intersectional_analysis", create_intersectional_analysis),
        ("counterfactual_analysis", create_counterfactual_analysis),
        ("score_distributions", create_score_distribution_analysis),
        ("comprehensive_summary", create_comprehensive_summary)
    ]
    
    for plot_type, plot_function in plot_functions:
        try:
            fig = plot_function(group_data, group)
            filename = f"{OUTPUT_DIR}/group_{group}_{plot_type}.png"
            fig.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"Saved {plot_type} for Group {group}")
        except Exception as e:
            print(f"Error creating {plot_type} for Group {group}: {str(e)}")

# Cross-group comparison
print("Creating cross-group comparison analysis...")

try:
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f"{PLOT_TITLE}: Cross-Group Comparison Analysis", fontsize=16, fontweight='bold')

    # Overall hire rates by group
    group_hire_rates = df_filtered.groupby('Group')['Binary Decision'].agg(['mean', 'std', 'count']).reset_index()
    bars = axes[0,0].bar(group_hire_rates['Group'], group_hire_rates['mean'], 
                        yerr=group_hire_rates['std'], capsize=4)
    axes[0,0].set_title('Hiring Rates Across Groups')
    axes[0,0].set_ylabel('Hiring Rate')
    
    # Add sample size labels
    for bar, count in zip(bars, group_hire_rates['count']):
        axes[0,0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                      f'n={count}', ha='center', va='bottom', fontsize=9)

    # Average competence scores by group
    group_competence = df_filtered.groupby('Group')['Competence Score (Int)'].agg(['mean', 'std', 'count']).reset_index()
    bars2 = axes[0,1].bar(group_competence['Group'], group_competence['mean'], 
                         yerr=group_competence['std'], capsize=4, color='lightcoral')
    axes[0,1].set_title('Average Competence Scores Across Groups')
    axes[0,1].set_ylabel('Average Competence Score')
    
    # Fairness metrics comparison across groups
    fairness_comparison = []
    for group in groups:
        group_data = df_filtered[df_filtered['Group'] == group]
        if len(group_data) > 10:  # Minimum sample size
            spd_metrics = calculate_statistical_parity_difference(group_data)
            di_metrics = calculate_disparate_impact(group_data)
            
            fairness_row = {'Group': group}
            fairness_row.update(spd_metrics)
            fairness_row.update(di_metrics)
            fairness_comparison.append(fairness_row)
    
    if fairness_comparison:
        fairness_df = pd.DataFrame(fairness_comparison)
        
        # Plot SPD metrics if available
        if 'Gender_SPD' in fairness_df.columns:
            axes[1,0].bar(fairness_df['Group'], fairness_df['Gender_SPD'], alpha=0.7)
            axes[1,0].set_title('Statistical Parity Difference (Gender) Across Groups')
            axes[1,0].set_ylabel('SPD Value')
            axes[1,0].axhline(y=0.1, color='red', linestyle='--', alpha=0.7, label='Threshold (0.1)')
            axes[1,0].legend()
        
        # Plot DI metrics if available
        if 'Gender_DI' in fairness_df.columns:
            bars3 = axes[1,1].bar(fairness_df['Group'], fairness_df['Gender_DI'], alpha=0.7, color='gold')
            axes[1,1].set_title('Disparate Impact Ratio (Gender) Across Groups')
            axes[1,1].set_ylabel('DI Ratio')
            axes[1,1].axhline(y=0.8, color='red', linestyle='--', alpha=0.7, label='Fair Threshold (0.8)')
            axes[1,1].legend()
            
            # Color bars based on fairness
            for bar, di_value in zip(bars3, fairness_df['Gender_DI']):
                if di_value < 0.8:
                    bar.set_color('red')
                    bar.set_alpha(0.7)

    plt.tight_layout()
    cross_group_filename = f"{OUTPUT_DIR}/cross_group_comparison.png"
    fig.savefig(cross_group_filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved cross-group comparison analysis")

except Exception as e:
    print(f"Error creating cross-group analysis: {str(e)}")

# Generate research insights report
print("Generating research insights report...")

try:
    insights_report = []
    insights_report.append("=" * 80)
    insights_report.append(f"HIRING BIAS ANALYSIS REPORT - {PLOT_TITLE}")
    insights_report.append("=" * 80)
    insights_report.append("")
    
    for group in groups:
        group_data = df_filtered[df_filtered['Group'] == group]
        if len(group_data) == 0:
            continue
            
        insights_report.append(f"GROUP {group} ANALYSIS")
        insights_report.append("-" * 40)
        insights_report.append(f"Sample Size: {len(group_data)}")
        insights_report.append(f"Overall Hiring Rate: {group_data['Binary Decision'].mean():.2%}")
        insights_report.append(f"Average Competence Score: {group_data['Competence Score (Int)'].mean():.2f}")
        insights_report.append("")
        
        # Gender analysis
        if len(group_data['Gender'].unique()) > 1:
            gender_analysis = group_data.groupby('Gender').agg({
                'Binary Decision': ['mean', 'count'],
                'Competence Score (Int)': ['mean', 'std']
            }).round(3)
            
            insights_report.append("GENDER ANALYSIS:")
            for gender in group_data['Gender'].unique():
                gender_subset = group_data[group_data['Gender'] == gender]
                hire_rate = gender_subset['Binary Decision'].mean()
                competence = gender_subset['Competence Score (Int)'].mean()
                count = len(gender_subset)
                insights_report.append(f"  {gender}: {hire_rate:.2%} hire rate, {competence:.2f} avg competence (n={count})")
        
        # Ethnicity analysis
        if len(group_data['Ethnicity'].unique()) > 1:
            insights_report.append("ETHNICITY ANALYSIS:")
            for ethnicity in group_data['Ethnicity'].unique():
                ethnicity_subset = group_data[group_data['Ethnicity'] == ethnicity]
                hire_rate = ethnicity_subset['Binary Decision'].mean()
                competence = ethnicity_subset['Competence Score (Int)'].mean()
                count = len(ethnicity_subset)
                insights_report.append(f"  {ethnicity}: {hire_rate:.2%} hire rate, {competence:.2f} avg competence (n={count})")
        
        # Fairness metrics
        spd_metrics = calculate_statistical_parity_difference(group_data)
        di_metrics = calculate_disparate_impact(group_data)
        stat_tests = perform_statistical_tests(group_data)
        
        if spd_metrics or di_metrics:
            insights_report.append("FAIRNESS METRICS:")
            for key, value in spd_metrics.items():
                insights_report.append(f"  Statistical Parity Difference ({key}): {value:.3f}")
            for key, value in di_metrics.items():
                fairness_status = "FAIR" if value >= 0.8 else "UNFAIR"
                insights_report.append(f"  Disparate Impact ({key}): {value:.3f} ({fairness_status})")
        
        if stat_tests:
            insights_report.append("STATISTICAL SIGNIFICANCE:")
            for key, value in stat_tests.items():
                significance = "***" if value < 0.001 else "**" if value < 0.01 else "*" if value < 0.05 else "ns"
                insights_report.append(f"  {key}: p={value:.4f} {significance}")
        
        insights_report.append("")
    
    # Overall insights
    insights_report.append("CROSS-GROUP INSIGHTS")
    insights_report.append("-" * 40)
    
    overall_hire_rate = df_filtered['Binary Decision'].mean()
    overall_competence = df_filtered['Competence Score (Int)'].mean()
    insights_report.append(f"Overall Dataset Hiring Rate: {overall_hire_rate:.2%}")
    insights_report.append(f"Overall Average Competence Score: {overall_competence:.2f}")
    
    # Group comparison
    group_stats = df_filtered.groupby('Group').agg({
        'Binary Decision': 'mean',
        'Competence Score (Int)': 'mean'
    }).round(3)
    
    insights_report.append("GROUP COMPARISON:")
    for group in group_stats.index:
        hire_rate = group_stats.loc[group, 'Binary Decision']
        competence = group_stats.loc[group, 'Competence Score (Int)']
        insights_report.append(f"  Group {group}: {hire_rate:.2%} hire rate, {competence:.2f} avg competence")
    
    # Research questions addressed
    insights_report.append("")
    insights_report.append("KEY RESEARCH QUESTIONS ADDRESSED:")
    insights_report.append("1. Statistical Parity Difference (SPD) - measures hiring rate differences")
    insights_report.append("2. Disparate Impact (DI) - 80% rule compliance check")
    insights_report.append("3. Equal Opportunity Difference - qualified candidate treatment")
    insights_report.append("4. Intersectional bias analysis - compound discrimination effects")
    insights_report.append("5. Counterfactual fairness - decision consistency analysis")
    insights_report.append("6. Score distribution analysis - competence evaluation patterns")
    insights_report.append("7. Statistical significance testing - confidence in observed differences")
    
    # Save report
    report_filename = f"{OUTPUT_DIR}/research_insights_report.txt"
    with open(report_filename, 'w') as f:
        f.write('\n'.join(insights_report))
    
    print(f"Saved research insights report to {report_filename}")
    
except Exception as e:
    print(f"Error generating insights report: {str(e)}")

print("All comprehensive research-focused analyses completed!")
print(f"All outputs saved to: {OUTPUT_DIR}")

# Print final summary
total_plots = len(groups) * 6 + 1  # 6 plot types per group + 1 cross-group comparison
print(f"Generated {total_plots} visualisation files addressing key research questions:")
print("   - Basic Distributions")
print("   - Fairness Metrics (SPD, DI, EOD)")
print("   - Intersectional Analysis")
print("   - Counterfactual Analysis")
print("   - Score Distribution Analysis")
print("   - Comprehensive Summary")
print("   - Cross-Group Comparison")
print("   - Research Insights Report")
print("")
print("Research Questions Addressed:")
print("   1. Name-only bias detection (Counterfactual Fairness)")
print("   2. Gender qualification bias analysis")
print("   3. Intersectional discrimination measurement")
print("   4. Statistical significance testing")
print("   5. Fairness metrics compliance (80% rule)")
print("   6. Score distribution pattern analysis")
print("   7. Cross-group consistency evaluation")
