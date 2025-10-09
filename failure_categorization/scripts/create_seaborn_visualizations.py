#!/usr/bin/env python3
"""
Create comprehensive seaborn visualizations from qualitative analysis results.
Generates research-quality charts with professional styling and multi-hue histograms.
"""

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from pathlib import Path
from datetime import datetime
import numpy as np

class SeabornAnalysisVisualizer:
    def __init__(self, results_dir: str, model_name: str | None = None):
        self.results_dir = Path(results_dir)
        self.model_name = model_name
        
        # Create timestamped output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = self.results_dir / f"seaborn_figures_{timestamp}"
        self.output_dir.mkdir(exist_ok=True)
        
        # Set beautiful styling
        plt.style.use('seaborn-v0_8')
        sns.set_theme(style="whitegrid", palette="deep")
        
        # Set global font properties for publication quality
        plt.rcParams.update({
            'font.size': 12,
            'font.family': 'serif',
            'axes.titlesize': 16,
            'axes.labelsize': 14,
            'xtick.labelsize': 11,
            'ytick.labelsize': 11,
            'legend.fontsize': 11,
            'figure.titlesize': 18,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'axes.grid': True,
            'grid.alpha': 0.3
        })
        
        print(f"🎨 Creating enhanced seaborn visualizations...")
        print(f"Output directory: {self.output_dir}")
    
    def get_failure_type_colors(self):
        """Professional color palette for failure types"""
        return {
            'wrong_solution': '#1f77b4',  # Deep blue
            'misunderstood_problem_statement': '#d62728',  # Deep red
            'tool_error': '#ff7f0e',  # Orange
            'infinite_loop': '#2ca02c',  # Green
            'endless_file_reading': '#9467bd',  # Purple
            'missed_edge_case': '#8c564b',  # Brown
            'syntax_error': '#e377c2',  # Pink
            'identified_incorrect_file': '#7f7f7f',  # Gray
            'exit_context': '#bcbd22',  # Olive
            'context_overflow_from_listing': '#17becf',  # Cyan
            'analysis_error': '#ffbb78',  # Light orange
            'exit_cost': '#98df8a',  # Light green
            'other': '#aec7e8',  # Light blue
        }
    
    def load_analysis_data(self):
        """Load and combine analysis data from JSON files"""
        print("Loading analysis data...")
        
        # Find all analysis JSON files (excluding format files)
        json_files = [f for f in self.results_dir.glob("*_analysis_*actions.json") 
                     if 'format' not in f.name.lower()]
        
        if not json_files:
            raise ValueError(f"No analysis JSON files found in {self.results_dir}")
        
        all_data = []
        
        for json_file in json_files:
            print(f"  Loading {json_file.name}...")
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # Extract category name from filename or data
            category = data.get('category', json_file.stem.replace('_analysis_20actions', ''))
            
            # Handle both old and new data structures
            # New structure has 'results', old structure has 'failed_instances'
            instances = data.get('results', data.get('failed_instances', []))
            print(f"  Loaded {len(instances)} instances from {category}")
            
            for instance in instances:
                # Handle different field names between structures
                failure_type = instance.get('failure_type', instance.get('category', 'other'))
                instance_id = instance.get('instance_id', 'unknown')
                reasoning = instance.get('reasoning', instance.get('description', ''))
                
                # Cost from token usage if available
                cost = instance.get('cost', 0.0)
                if cost == 0.0 and 'token_usage' in instance and instance['token_usage'] is not None:
                    cost = instance['token_usage'].get('total_cost', 0.0)
                
                all_data.append({
                    'exit_category': category,
                    'instance_id': instance_id,
                    'failure_type': failure_type,
                    'reasoning': reasoning,
                    'cost': cost
                })
        
        df = pd.DataFrame(all_data)
        print(f"Created DataFrame with {len(df)} rows covering {df['exit_category'].nunique()} exit categories")
        return df
    
    def create_proportional_stacked_bars(self, df):
        """Create proportional stacked bar chart matching the reference style"""
        print("Creating proportional stacked bar chart...")
        
        # Create pivot table
        pivot_data = df.pivot_table(
            values='instance_id', 
            index='exit_category', 
            columns='failure_type', 
            aggfunc='count', 
            fill_value=0
        )
        
        # Filter out failure types with zero total counts
        failure_type_totals = pivot_data.sum()
        non_zero_columns = failure_type_totals[failure_type_totals > 0].index
        pivot_data = pivot_data[non_zero_columns]
        
        # Get row totals for annotations
        row_totals = pivot_data.sum(axis=1)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Get colors - ensure consistent mapping
        colors = self.get_failure_type_colors()
        color_list = [colors.get(col, '#cccccc') for col in pivot_data.columns]
        
        # Create stacked bar chart
        bars = pivot_data.plot(kind='bar', stacked=True, ax=ax, color=color_list, 
                              width=0.7, edgecolor='white', linewidth=1.0)
        
        # Add count annotations above each bar
        for i, (idx, total) in enumerate(row_totals.items()):
            ax.text(i, total + max(row_totals) * 0.02, f'n={int(total)}', 
                   ha='center', va='bottom', fontweight='bold', fontsize=11)
        
        # Style the plot to match reference
        title_prefix = f"[{self.model_name}] " if self.model_name else ""
        ax.set_title(f"{title_prefix}Proportional Root Cause Distribution by Exit Category\n(Height proportional to count, segments show percentage within category)", 
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Exit Category', fontsize=12, fontweight='bold')
        ax.set_ylabel('Relative Scale (Height ∝ Instance Count)', fontsize=12, fontweight='bold')
        
        # Rotate x-axis labels
        plt.xticks(rotation=45, ha='right')
        
        # Add legend
        ax.legend(title='Failure Type', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        
        # Add grid
        ax.grid(axis='y', alpha=0.3, linestyle='-', linewidth=0.5)
        ax.set_axisbelow(True)
        
        # Set background
        ax.set_facecolor('white')
        fig.patch.set_facecolor('white')
        
        plt.tight_layout()
        
        # Save
        output_path = self.output_dir / "proportional_root_cause_distribution.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {output_path}")
        plt.close()
        
        return pivot_data
    
    def create_proportional_analysis(self, df):
        """Create proportional analysis with enhanced styling"""
        print("Creating proportional analysis chart...")
        
        # Create pivot table
        pivot_data = df.pivot_table(
            values='instance_id', 
            index='exit_category', 
            columns='failure_type', 
            aggfunc='count', 
            fill_value=0
        )
        
        # Filter out failure types with zero total counts
        failure_type_totals = pivot_data.sum()
        non_zero_columns = failure_type_totals[failure_type_totals > 0].index
        pivot_data = pivot_data[non_zero_columns]
        
        # Calculate proportions while maintaining absolute heights
        row_totals = pivot_data.sum(axis=1)
        proportions = pivot_data.div(row_totals, axis=0) * 100
        
        # Scale heights to be proportional to actual counts
        max_count = row_totals.max()
        scaled_data = pivot_data.copy().astype(float)  # Convert to float to avoid dtype warnings
        for idx in scaled_data.index:
            scale_factor = row_totals[idx] / max_count
            scaled_data.loc[idx] = scaled_data.loc[idx] * scale_factor
        
        # Create enhanced plot
        fig, ax = plt.subplots(figsize=(16, 10))
        
        # Get colors
        colors = self.get_failure_type_colors()
        color_list = [colors.get(col, '#cccccc') for col in scaled_data.columns]
        
        # Create stacked bar chart with enhanced styling
        bars = scaled_data.plot(kind='bar', stacked=True, ax=ax, color=color_list, 
                               width=0.7, edgecolor='white', linewidth=0.5)
        
        # Add instance count annotations
        for i, (idx, total) in enumerate(row_totals.items()):
            ax.text(i, total + max_count * 0.01, f'n={int(total)}', 
                   ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        # Beautify the plot
        title_prefix = f"[{self.model_name}] " if self.model_name else ""
        ax.set_title(f"{title_prefix}Proportional Root Cause Analysis by Exit Category\n(Heights proportional to instance counts)", 
                    fontsize=18, fontweight='bold', pad=30)
        ax.set_xlabel('Exit Category', fontsize=14, fontweight='bold')
        ax.set_ylabel('Scaled Instance Count', fontsize=14, fontweight='bold')
        
        plt.xticks(rotation=45, ha='right')
        
        # Enhanced legend (only add if there are legend elements)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            legend = ax.legend(title='Failure Type', bbox_to_anchor=(1.05, 1), 
                              loc='upper left', fontsize=11)
            legend.set_title('Failure Type', prop={'size': 12, 'weight': 'bold'})
        
        # Subtle grid
        ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
        ax.set_axisbelow(True)
        
        # Remove spines
        sns.despine()
        
        plt.tight_layout()
        
        # Save
        output_path = self.output_dir / "proportional_root_cause_analysis.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {output_path}")
        plt.close()
        
        return scaled_data, proportions
    
    def create_failure_type_proportions_heatmap(self, df):
        """Create percentage heatmap matching the reference style"""
        print("Creating failure type proportions heatmap...")
        
        # Create pivot table
        pivot_data = df.pivot_table(
            values='instance_id', 
            index='exit_category', 
            columns='failure_type', 
            aggfunc='count', 
            fill_value=0
        )
        
        # Convert to percentages within each row (exit category)
        percentage_data = pivot_data.div(pivot_data.sum(axis=1), axis=0) * 100
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Use yellow-red colormap like the reference
        sns.heatmap(percentage_data, 
                   annot=True, 
                   fmt='.1f', 
                   cmap='YlOrRd',  # Yellow to red colormap like reference
                   cbar_kws={'label': 'Percentage of Category', 'shrink': 0.8},
                   square=False,
                   linewidths=0.5,
                   linecolor='white',
                   vmin=0,
                   vmax=100,
                   ax=ax)
        
        # Style to match reference
        title_prefix = f"[{self.model_name}] " if self.model_name else ""
        ax.set_title(f"{title_prefix}Failure Type Proportions by Exit Category\n(Percentage within each category)", 
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Failure Type', fontsize=12, fontweight='bold')
        ax.set_ylabel('Exit Category', fontsize=12, fontweight='bold')
        
        # Rotate labels
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        
        # Save
        output_path = self.output_dir / "failure_type_proportions_heatmap.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {output_path}")
        plt.close()
        
        return percentage_data
    
    def create_overall_failure_type_distribution(self, df):
        """Create horizontal bar chart matching the reference style"""
        print("Creating overall failure type distribution...")
        
        # Count failure types and sort by count (descending)
        failure_counts = df['failure_type'].value_counts().sort_values(ascending=True)  # ascending for horizontal bars
        
        # Create horizontal bar chart
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Get colors
        colors = self.get_failure_type_colors()
        color_list = [colors.get(ft, '#cccccc') for ft in failure_counts.index]
        
        # Create horizontal bar plot
        bars = ax.barh(failure_counts.index, failure_counts.values, 
                      color=color_list, alpha=0.8, edgecolor='white', linewidth=0.5)
        
        # Add value labels on bars
        for i, (failure_type, count) in enumerate(failure_counts.items()):
            ax.text(count + max(failure_counts.values) * 0.01, i, str(count), 
                   ha='left', va='center', fontweight='bold', fontsize=11)
        
        # Style to match reference
        title_prefix = f"[{self.model_name}] " if self.model_name else ""
        ax.set_title(f"{title_prefix}Overall Failure Type Distribution\n(Total count across all exit categories)", 
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Total Occurrences', fontsize=12, fontweight='bold')
        ax.set_ylabel('Failure Type', fontsize=12, fontweight='bold')
        
        # Add grid
        ax.grid(axis='x', alpha=0.3, linestyle='-', linewidth=0.5)
        ax.set_axisbelow(True)
        
        # Set background
        ax.set_facecolor('white')
        fig.patch.set_facecolor('white')
        
        # Remove spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        
        # Save
        output_path = self.output_dir / "overall_failure_type_distribution.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {output_path}")
        plt.close()
        
        # Also export the plotted data to CSV (counts and percentages)
        try:
            counts_desc = failure_counts.sort_values(ascending=False)
            csv_df = counts_desc.rename_axis('failure_type').reset_index(name='count')
            total = csv_df['count'].sum() or 1
            csv_df['percentage'] = (csv_df['count'] / total).round(6)
            csv_path = self.output_dir / "overall_failure_type_distribution.csv"
            csv_df.to_csv(csv_path, index=False)
            print(f"Saved: {csv_path}")
        except Exception as e:
            print(f"Warning: failed to write overall_failure_type_distribution.csv: {e}")

        return failure_counts
    
    def create_cost_analysis(self, df):
        """Create cost distribution analysis"""
        print("Creating cost analysis...")
        
        # Only create if cost data exists
        if 'cost' not in df.columns or df['cost'].sum() == 0:
            print("No cost data available, skipping cost analysis...")
            return
        
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            
            # Build a safe palette only for present failure types
            present_types = sorted([t for t in df['failure_type'].dropna().unique().tolist()])
            base_palette = self.get_failure_type_colors()
            safe_palette = {t: base_palette.get(t, '#cccccc') for t in present_types}

            # Cost distribution by failure type
            sns.boxplot(
                data=df,
                y='failure_type',
                x='cost',
                hue='failure_type',
                palette=safe_palette,
                legend=False,
                ax=ax1
            )
            ax1.set_title('Cost Distribution by Failure Type', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Analysis Cost ($)', fontsize=12, fontweight='bold')
            ax1.set_ylabel('Failure Type', fontsize=12, fontweight='bold')
            
            # Overall cost histogram
            sns.histplot(data=df, x='cost', bins=20, kde=True, color='steelblue', alpha=0.7, ax=ax2)
            ax2.set_title('Overall Cost Distribution', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Analysis Cost ($)', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Count', fontsize=12, fontweight='bold')
            
            # Styling
            for ax in [ax1, ax2]:
                ax.grid(alpha=0.3, linestyle='--', linewidth=0.5)
                ax.set_axisbelow(True)
            
            sns.despine()
            plt.tight_layout()
            
            # Save
            output_path = self.output_dir / "cost_analysis.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"Saved: {output_path}")
            plt.close()
        except Exception as e:
            print(f"Skipping cost analysis due to error: {e}")
    
    def generate_summary_report(self, df):
        """Generate enhanced summary report"""
        summary_stats = {
            "total_instances": len(df),
            "exit_categories": df['exit_category'].nunique(),
            "failure_types": df['failure_type'].nunique(),
            "total_cost": df['cost'].sum() if 'cost' in df.columns else 0,
            "avg_cost": df['cost'].mean() if 'cost' in df.columns else 0,
        }
        
        # Category breakdown
        category_breakdown = df['exit_category'].value_counts().to_dict()
        failure_breakdown = df['failure_type'].value_counts().to_dict()
        
        # Generate enhanced report
        report_content = f"""
# 📊 SWE Agent Failure Analysis Summary Report
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## 📈 Overall Statistics
- **Total Instances Analyzed**: {summary_stats['total_instances']:,}
- **Exit Categories**: {summary_stats['exit_categories']}
- **Failure Types Identified**: {summary_stats['failure_types']}
- **Total Analysis Cost**: ${summary_stats['total_cost']:.4f}
- **Average Cost per Instance**: ${summary_stats['avg_cost']:.4f}

## 🎯 Exit Category Breakdown
"""
        for category, count in category_breakdown.items():
            percentage = (count / summary_stats['total_instances']) * 100
            report_content += f"- **{category}**: {count} instances ({percentage:.1f}%)\n"
        
        report_content += "\n## 🔍 Failure Type Breakdown\n"
        for failure_type, count in failure_breakdown.items():
            percentage = (count / summary_stats['total_instances']) * 100
            report_content += f"- **{failure_type}**: {count} instances ({percentage:.1f}%)\n"
        
        # Save report
        # Prepend model name to summary if provided
        if self.model_name:
            header = f"Model: {self.model_name}\n\n"
        else:
            header = ""
        report_path = self.output_dir / "analysis_summary.txt"
        with open(report_path, 'w') as f:
            f.write(header + report_content)
        
        print(f"Saved: {report_path}")
        return summary_stats
    
    def create_all_visualizations(self, df):
        """Create all visualizations matching the reference style"""
        print("\n🎨 Generating visualizations to match reference style...\n")
        
        # 1. Proportional stacked bars (matches first reference image)
        pivot_data = self.create_proportional_stacked_bars(df)
        
        # 2. Percentage heatmap (matches second reference image)  
        percentage_data = self.create_failure_type_proportions_heatmap(df)
        
        # 3. Horizontal bar chart (matches third reference image)
        failure_counts = self.create_overall_failure_type_distribution(df)
        
        # Optional: Keep cost analysis (if data exists)
        self.create_cost_analysis(df)
        
        # Summary report
        summary_stats = self.generate_summary_report(df)
        
        return summary_stats

def main():
    parser = argparse.ArgumentParser(description="Create enhanced seaborn visualizations")
    parser.add_argument("--results_dir", required=True, help="Directory containing analysis results")
    parser.add_argument("--model_name", help="Human-readable model name for titles and reports")
    
    args = parser.parse_args()
    
    try:
        # Create visualizer
        visualizer = SeabornAnalysisVisualizer(args.results_dir, model_name=args.model_name)
        
        # Load data
        df = visualizer.load_analysis_data()
        
        # Create all visualizations
        summary_stats = visualizer.create_all_visualizations(df)
        
        print(f"\n{'='*60}")
        print("✅ ENHANCED SEABORN VISUALIZATION COMPLETE!")
        print(f"{'='*60}")
        print(f"📁 All files saved to: {visualizer.output_dir}")
        print(f"\n📊 Generated visualizations matching reference style:")
        print("• proportional_root_cause_distribution.png - PROPORTIONAL STACKED BARS")
        print("• failure_type_proportions_heatmap.png - PERCENTAGE HEATMAP")
        print("• overall_failure_type_distribution.png - HORIZONTAL BAR CHART")
        print("• cost_analysis.png - COST ANALYSIS (if available)")
        print("• analysis_summary.txt - DETAILED REPORT")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())