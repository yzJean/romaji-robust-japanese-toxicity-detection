#!/bin/bash
# Compute Error Taxonomy (Type B / Type C ratio) for Section 7.2.3
# Run this after you have generated the CSV results from inference.py
#
# Usage:
#   ./analyze_errors.sh file1.csv [file2.csv] [file3.csv] ...
#   ./analyze_errors.sh outputs/eval/bert_romaji_results.csv outputs/eval/mdeberta_romaji_results.csv
#   ./analyze_errors.sh outputs/eval/*.csv

set -e

# Function to display usage
show_usage() {
    echo "Usage: $0 <csv_file1> [csv_file2] [csv_file3] ..."
    echo ""
    echo "Examples:"
    echo "  $0 outputs/eval/bert_romaji_results.csv"
    echo "  $0 outputs/eval/bert_romaji_results.csv outputs/eval/mdeberta_romaji_results.csv"
    echo "  $0 outputs/eval/bert_romaji_results.csv outputs/eval/mdeberta_romaji_results.csv outputs/eval/byt5_romaji_results.csv"
    echo "  $0 outputs/eval/*_romaji_results.csv"
    echo ""
    echo "Default behavior (no arguments): Analyze bert and mdeberta results"
    exit 1
}

echo "=========================================="
echo "Section 7.2.3 Error Taxonomy Analysis"
echo "Computing Type B / Type C Ratio"
echo "=========================================="
echo ""

# Check for help flag
if [ "$1" == "--help" ] || [ "$1" == "-h" ]; then
    show_usage
fi

# If no arguments provided, use default files
if [ $# -eq 0 ]; then
    echo "No arguments provided. Using default files..."
    CSV_FILES=(
        "outputs/eval/bert_romaji_results.csv"
        "outputs/eval/mdeberta_romaji_results.csv"
    )
else
    # Use provided arguments
    CSV_FILES=("$@")
fi

# Validate that all files exist
MISSING_FILES=0
for file in "${CSV_FILES[@]}"; do
    if [ ! -f "$file" ]; then
        echo "Error: File not found: $file"
        MISSING_FILES=$((MISSING_FILES + 1))
    fi
done

if [ $MISSING_FILES -gt 0 ]; then
    echo ""
    echo "Error: $MISSING_FILES file(s) not found!"
    echo "Please ensure all CSV files exist before running this script."
    echo ""
    show_usage
fi

# Display files to be analyzed
echo "Analyzing ${#CSV_FILES[@]} file(s):"
for file in "${CSV_FILES[@]}"; do
    echo "  - $file"
done
echo ""

# Determine output filename based on number of files
if [ ${#CSV_FILES[@]} -eq 1 ]; then
    OUTPUT_FILE="outputs/eval/error_taxonomy_summary.json"
else
    OUTPUT_FILE="outputs/eval/error_taxonomy_comparison.json"
fi

# Run analysis
python src/compute_error_taxonomy.py \
    "${CSV_FILES[@]}" \
    --output "$OUTPUT_FILE"

echo ""
echo "=========================================="
echo "✓ Analysis complete!"
echo "=========================================="
echo ""
echo "Analyzed ${#CSV_FILES[@]} model(s)"
echo "Summary saved to: $OUTPUT_FILE"
echo ""
