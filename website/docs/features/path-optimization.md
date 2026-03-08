# Path Optimization

Path optimization reorders cutting segments to minimize travel distance. The laser moves efficiently between cuts instead of jumping randomly across the work area.

## How It Works

Without optimization, paths are cut in the order they appear in your design file. Optimization analyzes all path segments and rearranges them so the laser travels the shortest total distance between cuts.

**Before optimization:** Laser jumps back and forth across the material  
**After optimization:** Laser moves sequentially from cut to cut

## Settings

### Enable Optimization

Toggle path optimization on or off. Enabled by default for most operations.

## When to Use Optimization

**Enable for:**

- Designs with many separate shapes
- Reducing total job time
- Minimizing wear on motion system
- Complex nested layouts

**Disable for:**

- Designs where cut order matters (e.g., inside features before outside)
- Debugging path issues
- When you need predictable, repeatable execution order

## How It Affects Your Job

**Time savings:** Can reduce job time by 20-50% for designs with many separate cuts.

**Motion efficiency:** Less rapid movement means less wear on belts, motors, and bearings.

**Heat distribution:** Optimized paths may concentrate heat in one area. For heat-sensitive materials, consider if order matters.

:::tip
Optimization runs automatically. Just enable it and the software handles the rest.
:::

---

## Related Topics

- [Contour Cutting](operations/contour) - Primary cutting operation
- [Holding Tabs](holding-tabs) - Keeping parts secure during cutting
