#!/bin/bash

# Config
DUCK_DECODER="python3 duck_decoder.py"  # Adjust if needed
LOG_FILE="conversion_log.txt"
FAILED_FILES="failed_conversions.txt"

# Initialize log files
echo "Starting PNG to MP4 conversion at $(date)" | tee -a "$LOG_FILE"
echo "" > "$FAILED_FILES"  # Clear previous failures log

# Process each PNG file
for png_file in *.png; do
    [ -f "$png_file" ] || continue  # Skip if not a file

    mp4_file="${png_file%.*}.mp4"  # Generate output name
    
    echo "Converting: $png_file → $mp4_file" | tee -a "$LOG_FILE"

    # Attempt conversion
    if $DUCK_DECODER --duck "$png_file" --out "$mp4_file"; then
        # Verify MP4 was created before deletion
        if [ -f "$mp4_file" ]; then
            rm -v "$png_file" | tee -a "$LOG_FILE"
            echo "✅ Success: $png_file converted and deleted" | tee -a "$LOG_FILE"
        else
            echo "❌ ERROR: $mp4_file not generated! Skipping." | tee -a "$LOG_FILE"
            echo "$png_file" >> "$FAILED_FILES"
        fi
    else
        echo "❌ Conversion FAILED for $png_file. Skipping." | tee -a "$LOG_FILE"
        echo "$png_file" >> "$FAILED_FILES"
    fi

    echo "---------------------------------" | tee -a "$LOG_FILE"
done

# Completion summary
echo ""
echo "================================="
echo "Conversion complete!"
echo " - Full log: $LOG_FILE"
echo " - Failed files: $FAILED_FILES"
echo " - $(date)"
echo "================================="

