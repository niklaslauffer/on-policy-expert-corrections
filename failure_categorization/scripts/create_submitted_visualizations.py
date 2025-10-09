#!/usr/bin/env python3
"""
Create comprehensive seaborn visualizations from submitted solution analysis results.
Generates research-quality charts focused on solution-oriented failure categories.
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

class SubmittedAnalysisVisualizer:
    def __init__(self, results_dir: str):
        self.results_dir = Path(results_dir)
        
        # Create timestamped output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = self.results_dir / f"submitted_figures_{timestamp}"
        self.output_dir.mkdir(exist_ok=True)
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        print(f"🎨 Creating submitted case solution analysis visualizations...")
        print(f"Output directory: {self.output_dir}")
        
    def get_solution_failure_colors(self):
        """Define consistent, professional color mapping for solution-oriented failure types"""
        return {
            'issue_replication': '#1f77b4',      # Deep blue
            'solution_confirmation': '#ff7f0e',  # Orange
            'logical_clarity': '#2ca02c',        # Green
            'hallucinations': '#d62728',         # Red
            'consistency': '#9467bd',            # Purple
            'efficiency': '#8c564b'              # Brown
        }
    
    def load_analysis_data(self):
        """Load all submitted analysis JSON files from the results directory"""
        analysis_files = list(self.results_dir.glob("*submitted_analysis_*actions.json"))
        
        if not analysis_files:
            raise ValueError(f"No submitted analysis JSON files found in {self.results_dir}")
        
        print(f"Loading submitted analysis data...")
        print(f"Found {len(analysis_files)} submitted analysis files")
        
        all_data = []
        
        for file_path in analysis_files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                category = data.get('category', file_path.stem.replace('_submitted_analysis_20actions', ''))
                results = data.get('results', [])
                
                for result in results:
                    all_data.append({
                        'exit_category': category,
                        'instance_id': result.get('instance_id'),
                        'failure_type': result.get('failure_type', 'logical_clarity'),
                        'confidence': result.get('confidence', 0.5),
                        'reasoning': result.get('reasoning', '')
                    })
                    
                print(f"  Loaded {len(results)} results from {category}")
                
            except Exception as e:
                print(f"  Error loading {file_path}: {e}")
                continue
        
        if not all_data:
            raise ValueError("No valid data found in submitted analysis files")
        
        df = pd.DataFrame(all_data)
        print(f"Created DataFrame with {len(df)} rows covering {df['exit_category'].nunique()} exit categories")
        
        return df
    
    def create_solution_failure_by_exit_category(self, df):
        """Create stacked bar chart of solution failures by exit category"""
        print("Creating solution failure by exit category visualization...")
        
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
        
        # Get colors
        colors = self.get_solution_failure_colors()
        color_list = [colors.get(col, '#cccccc') for col in pivot_data.columns]
        
        # Create plot
        fig, ax = plt.subplots(figsize=(14, 8))
        
        pivot_data.plot(kind='bar', stacked=True, ax=ax, color=color_list, width=0.8)
        
        plt.title('Solution Process Failure Analysis by Exit Category', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Exit Category', fontsize=12, fontweight='bold')
        plt.ylabel('Number of Instances', fontsize=12, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Solution Failure Type', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        # Save
        output_path = self.output_dir / "solution_failure_by_exit_category.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
        plt.close()
        
        return pivot_data
    
    def create_proportional_solution_failure_by_category(self, df):
        """Create proportional stacked bar chart with heights proportional to instance counts"""
        print("Creating proportional solution failure chart...")
        
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
        
        # Calculate proportions for each category
        proportions = pivot_data.div(pivot_data.sum(axis=1), axis=0)
        
        # Get total counts for each category (for bar heights)
        total_counts = pivot_data.sum(axis=1)
        
        # Scale proportions by total counts to get proportional heights
        scaled_data = proportions.multiply(total_counts, axis=0)
        
        # Get colors
        colors = self.get_solution_failure_colors()
        color_list = [colors.get(col, '#cccccc') for col in scaled_data.columns]
        
        # Create plot
        fig, ax = plt.subplots(figsize=(14, 8))
        
        scaled_data.plot(kind='bar', stacked=True, ax=ax, color=color_list, width=0.8)
        
        # Add instance count annotations
        for i, (category, count) in enumerate(total_counts.items()):
            ax.text(i, count + max(total_counts) * 0.01, f'n={count}', 
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        plt.title('Proportional Solution Failure Analysis by Exit Category\n(Bar heights proportional to instance counts)', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Exit Category', fontsize=12, fontweight='bold')
        plt.ylabel('Number of Instances (Proportional)', fontsize=12, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Solution Failure Type', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        # Save
        output_path = self.output_dir / "proportional_solution_failure_by_category.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
        plt.close()
        
        return scaled_data, proportions
    
    def create_solution_failure_heatmap(self, df):
        """Create heatmap of solution failure types across exit categories"""
        print("Creating solution failure heatmap...")
        
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
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        sns.heatmap(pivot_data, annot=True, fmt='d', cmap='Blues', 
                   cbar_kws={'label': 'Number of Instances'}, ax=ax)
        
        plt.title('Solution Failure Type Distribution Heatmap', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Solution Failure Type', fontsize=12, fontweight='bold')
        plt.ylabel('Exit Category', fontsize=12, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        # Save
        output_path = self.output_dir / "solution_failure_heatmap.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
        plt.close()
    
    def create_solution_failure_overview(self, df):
        """Create overview chart of solution failure types across all categories"""
        print("Creating solution failure type overview...")
        
        # Count failure types
        failure_counts = df['failure_type'].value_counts()
        
        # Get colors
        colors = self.get_solution_failure_colors()
        color_list = [colors.get(failure_type, '#cccccc') for failure_type in failure_counts.index]
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        bars = failure_counts.plot(kind='bar', ax=ax, color=color_list, width=0.8)
        
        plt.title('Overall Solution Failure Type Distribution', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Solution Failure Type', fontsize=12, fontweight='bold')
        plt.ylabel('Number of Instances', fontsize=12, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar in bars.patches:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max(failure_counts) * 0.01,
                   f'{int(height)}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        
        # Save
        output_path = self.output_dir / "solution_failure_overview.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
        plt.close()
    
    def generate_summary_report(self, df):
        """Generate text summary of the submitted analysis"""
        print("Generating summary report...")
        
        summary_lines = []
        summary_lines.append("SUBMITTED SOLUTION ANALYSIS SUMMARY")
        summary_lines.append("=" * 50)
        summary_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        summary_lines.append(f"Total submitted instances analyzed: {len(df)}")
        summary_lines.append(f"Exit categories: {df['exit_category'].nunique()}")
        summary_lines.append(f"Solution failure types: {df['failure_type'].nunique()}")
        summary_lines.append("")
        
        # Exit category breakdown
        summary_lines.append("SUBMITTED EXIT CATEGORY BREAKDOWN:")
        category_counts = df['exit_category'].value_counts()
        for category, count in category_counts.items():
            percentage = (count / len(df)) * 100
            summary_lines.append(f"  {category}: {count} instances ({percentage:.1f}%)")
        summary_lines.append("")
        
        # Solution failure type breakdown
        summary_lines.append("SOLUTION FAILURE TYPE BREAKDOWN:")
        failure_counts = df['failure_type'].value_counts()
        for failure_type, count in failure_counts.items():
            percentage = (count / len(df)) * 100
            summary_lines.append(f"  {failure_type}: {count} instances ({percentage:.1f}%)")
        summary_lines.append("")
        
        # Confidence analysis
        avg_confidence = df['confidence'].mean()
        summary_lines.append(f"CONFIDENCE ANALYSIS:")
        summary_lines.append(f"  Average confidence: {avg_confidence:.3f}")
        summary_lines.append(f"  High confidence (>0.8): {len(df[df['confidence'] > 0.8])} instances")
        summary_lines.append(f"  Low confidence (<0.5): {len(df[df['confidence'] < 0.5])} instances")
        
        # Save summary
        summary_text = "\n".join(summary_lines)
        output_path = self.output_dir / "submitted_analysis_summary.txt"
        with open(output_path, 'w') as f:
            f.write(summary_text)
        
        print(f"Saved: {output_path}")
        return summary_text
    
    def create_all_visualizations(self):
        """Create all visualization types for submitted analysis"""
        # Load data
        df = self.load_analysis_data()
        
        # Create visualizations
        pivot_data = self.create_solution_failure_by_exit_category(df)
        scaled_data, proportions = self.create_proportional_solution_failure_by_category(df)
        self.create_solution_failure_heatmap(df)
        self.create_solution_failure_overview(df)
        
        # Generate summary
        summary = self.generate_summary_report(df)
        
        return df, summary

def main():
    parser = argparse.ArgumentParser(description="Create submitted solution analysis visualizations")
    parser.add_argument("--results_dir", required=True, 
                       help="Directory containing submitted analysis JSON files")
    
    args = parser.parse_args()
    
    try:
        visualizer = SubmittedAnalysisVisualizer(args.results_dir)
        df, summary = visualizer.create_all_visualizations()
        
        print("=" * 60)
        print("✅ SUBMITTED SOLUTION ANALYSIS VISUALIZATIONS COMPLETE!")
        print("=" * 60)
        print(f"📁 All files saved to: {visualizer.output_dir}")
        print("📊 Generated visualizations:")
        print("• solution_failure_by_exit_category.png - PRIMARY CHART")
        print("• proportional_solution_failure_by_category.png - PROPORTIONAL COMPARISON") 
        print("• solution_failure_heatmap.png")
        print("• solution_failure_overview.png")
        print("• submitted_analysis_summary.txt")
        
    except Exception as e:
        print(f"❌ Error creating submitted visualizations: {e}")
        raise

if __name__ == "__main__":
    main()
