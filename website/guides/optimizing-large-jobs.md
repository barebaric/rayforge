# Optimizing Large Jobs

Learn techniques for improving performance and efficiency when working with complex designs and large-scale laser cutting projects.

## Overview

Large jobs with thousands of vector paths or high-resolution rasters can strain system resources and result in long processing times. This guide covers optimization strategies.

## Common Performance Issues

- Long file loading times
- Slow preview rendering
- G-code generation delays
- Machine buffering issues
- Jerky motion during execution

## Design Optimization

### Simplify Vector Paths

- **Reduce node count**: Use fewer anchor points where possible
- **Combine overlapping paths**: Merge duplicate or overlapping vectors
- **Remove hidden geometry**: Delete elements that won't be cut
- **Optimize curves**: Simplify complex curves while maintaining quality

### Raster Optimization

- **Reduce resolution**: 300-500 DPI is often sufficient for engraving
- **Crop images tightly**: Remove excess whitespace
- **Use appropriate dither**: Choose dithering algorithm based on content
- **Optimize contrast**: Pre-process images for better conversion

### Layer Management

- **Group similar operations**: Combine operations where power/speed match
- **Order operations logically**: Cut inner details before outer perimeters
- **Use layer colors strategically**: Assign distinct colors for easy organization

## File Preparation

### Before Importing

- **Clean up source files** in vector editor (Inkscape, Illustrator, etc.)
- **Convert text to paths** to avoid font issues
- **Simplify complex designs** before import
- **Scale appropriately** for your machine

### Import Settings

- **Choose appropriate DPI** for rasters (300-500 DPI typically)
- **Set correct units** (mm, inches) before import
- **Verify dimensions** after import
- **Check path direction** for efficient cutting order

## Operation Configuration

### Path Optimization

Rayforge can optimize cutting order for efficiency:

1. **Enable path optimization** in operation settings
2. **Choose optimization method**:
   - **Nearest neighbor**: Fast, good for scattered elements
   - **Greedy TSP**: Better optimization, longer calculation
3. **Set optimization constraints** (inside-out, by layer, etc.)

### Speed and Acceleration

Balance speed with quality:

- **Higher speeds**: Faster job completion but may affect quality
- **Lower acceleration**: Smoother motion, less vibration
- **Optimize for material**: Test settings for each material type

### Raster Settings

- **Increase line spacing**: If quality allows, wider spacing = faster jobs
- **Optimize overscan**: Minimal overscan reduces travel time
- **Bidirectional engraving**: Enable for faster rasters
- **Adjust DPI**: Lower resolution for less critical details

## Machine Configuration

### GRBL Settings

Optimize controller settings for performance:

- **Increase buffer size** if supported
- **Adjust acceleration** ($120-$122) for smooth motion
- **Optimize max feed rate** ($110-$112) for your machine
- **Enable features** like laser mode ($32=1)

See [GRBL Settings](../machine/grbl-settings.md) for details.

### Connection Settings

- **Use reliable USB connection** (avoid hubs if possible)
- **Check baud rate**: Higher rates (115200) for better throughput
- **Monitor buffer usage**: Adjust if underruns occur
- **Use dedicated USB port**: Avoid sharing with other devices

## Workflow Strategies

### Break Large Jobs into Segments

For very large projects:

1. **Divide into sections** that fit comfortably in memory
2. **Process sections independently**
3. **Use registration marks** for alignment between sections
4. **Verify alignment** before committing to cuts

### Use Simulation Mode

Preview before cutting:

1. **Run simulation** to verify motion and timing
2. **Check for issues**: Overlaps, gaps, unexpected moves
3. **Estimate job time** accurately
4. **Adjust settings** based on simulation results

See [Simulation Mode](../features/simulation-mode.md) for details.

### Progressive Testing

Don't commit to full runs without testing:

1. **Test on scrap material** first
2. **Run small section** to verify settings
3. **Check quality** before full job
4. **Adjust and repeat** until satisfied

## System Optimization

### Computer Performance

- **Close unnecessary applications** during job processing
- **Disable power saving** that might throttle CPU
- **Use SSD storage** for better file I/O
- **Ensure adequate RAM** (8GB minimum, 16GB+ recommended)

### Software Settings

- **Adjust preview quality**: Lower quality for faster interaction
- **Disable real-time preview** during editing if slow
- **Increase undo history** if needed, or reduce for less memory use
- **Configure auto-save** to prevent loss without impacting performance

## Monitoring and Troubleshooting

### During Execution

- **Monitor progress**: Watch for unusual behavior
- **Check temperature**: Ensure laser and motors don't overheat
- **Verify motion**: Look for missed steps or binding
- **Listen for issues**: Unusual sounds may indicate problems

### Common Issues

**Jerky motion**: Reduce acceleration or increase buffer size

**Long pauses**: Check for communication issues or buffer underruns

**Missed steps**: Reduce speed or acceleration, check mechanical issues

**Out of memory**: Simplify design or process in segments

## Benchmarking

Track performance improvements:

- **Note job completion times** before and after optimization
- **Document settings** that work well for specific job types
- **Share findings** with the community
- **Maintain a testing log** for reference

## Related Topics

- [Simulation Mode](../features/simulation-mode.md)
- [GRBL Settings](../machine/grbl-settings.md)
- [Multi-Layer Workflow](../features/multi-layer.md)
- [Batch Production Workflow](batch-production-workflow.md)
