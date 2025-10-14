# Stock Handling Workflow

Stock handling in Rayforge is a sequential process that allows you to define the physical material you'll be working with, assign properties to it, and then organize your design elements on it. This guide walks you through the complete workflow from adding stock to auto-layouting your design.

## 1. Adding Stock

Stock represents the physical piece of material you'll be cutting or engraving. To add stock to your document:

1. In the **Stock Material** panel in the left sidebar, click the **Add Stock** button
2. A new stock item will be created with default dimensions (80% of your machine's workspace)
3. The stock will appear as a rectangle in the workspace, centered on the machine bed

### Stock Properties

Each stock item has the following properties:
- **Name**: A descriptive name for identification (auto-numbered as "Stock 1", "Stock 2", etc.)
- **Dimensions**: Width and height of the stock material
- **Thickness**: The material thickness (optional but recommended for accurate 3D preview)
- **Material**: The type of material (assigned in the next step)
- **Visibility**: Toggle to show/hide the stock in the workspace

### Managing Stock Items

- **Rename**: Open the Stock Properties dialog and edit the name field
- **Resize**: Select the stock item in the workspace and drag the corner handles to resize
- **Move**: Select the stock item in the workspace and drag to reposition it
- **Delete**: Click the delete button (trash icon) next to the stock item in the Stock Material panel
- **Edit properties**: Click the properties button (document icon) to open the Stock Properties dialog
- **Toggle visibility**: Click the visibility button (eye icon) to show/hide the stock item

## 2. Assigning Material

Once you have stock defined, you can assign a material to it:

1. In the **Stock Material** panel, click the properties button (document icon) on the stock item
2. In the Stock Properties dialog, click the **Select** button next to the Material field
3. Browse through your material libraries and select the appropriate material
4. The stock will update to show the material's visual appearance

### Material Properties

Materials define the visual properties of your stock:
- **Visual appearance**: Color and pattern for visualization
- **Category**: Grouping (e.g., "Wood", "Acrylic", "Metal")
- **Description**: Additional information about the material

Note: Material properties are defined in material libraries and cannot be edited through the stock properties dialog. The stock properties only allow you to assign a material to a stock item.

## 3. Assigning Stock to Layers

After defining your stock and assigning materials, you can associate layers with specific stock items:

1. In the **Layers** panel, locate the layer you want to assign to stock
2. Click the stock assignment button (shows "Whole Surface" by default)
3. From the dropdown menu, select the stock item you want to associate with this layer
4. The content of that layer will now be constrained to the boundaries of the assigned stock

You can also choose "Whole Surface" to use the entire machine workspace instead of a specific stock item.

### Why Assign Stock to Layers?

- **Layout boundaries**: Provides boundaries for the auto-layout algorithm to work within
- **Visual organization**: Helps organize your design by associating layers with physical materials
- **Material visualization**: Shows the visual appearance of the assigned material on the stock

## 4. Auto-Layout

The auto-layout feature helps you efficiently arrange your design elements:

1. Select the items you want to arrange (or leave nothing selected to arrange all items in the active layer)
2. Click the **Arrange** button in the toolbar and select **Auto Layout (pack workpieces)**
3. Rayforge will automatically arrange the items to optimize material usage

### Auto-Layout Behavior

The auto-layout algorithm works differently depending on your layer configuration:

- **If a stock item is assigned to the layer**: Items are arranged within the boundaries of that specific stock item
- **If "Whole Surface" is selected**: Items are arranged across the entire machine workspace

The algorithm considers:
- **Item boundaries**: Respects the dimensions of each design element
- **Rotation**: Can rotate items in 90-degree increments for better fit
- **Spacing**: Maintains a margin between items (default 0.5mm)
- **Stock boundaries**: Keeps all items within the defined boundaries

### Manual Layout Alternatives

If you prefer more control, Rayforge also offers manual layout tools:
- **Alignment tools**: Align left, right, center, top, bottom
- **Distribution tools**: Spread items horizontally or vertically
- **Individual positioning**: Click and drag items to place them manually

## Tips for Effective Stock Handling

1. **Start with accurate stock dimensions** - Measure your material precisely for best results
2. **Use descriptive names** - Name your stock items clearly (e.g., "Birch Plywood 3mm")
3. **Set material thickness** - This can be useful for future calculations and reference
4. **Assign materials early** - This ensures proper visual representation from the start
5. **Use layers for organization** - Separate different parts of your design into layers before assigning to stock
6. **Check fit before cutting** - Use the 2D view to verify everything fits on your stock material

## Troubleshooting

### Auto-layout doesn't work as expected
- Check if your layer has a stock assigned
- Make sure items are not grouped (ungroup them first)
- Try reducing the number of items selected at once
- Verify that items fit within the boundaries (stock or whole surface)
