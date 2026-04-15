---
description: "Manage stock material dimensions and positioning in Rayforge. Define workpiece size, origin, and alignment for consistent laser cutting results."
---

# Stock Handling

Stock in Rayforge represents the physical material you'll be cutting or engraving. Stock is a **document-global** concept—your document can have one or more stock items, and they exist independently of layers.

## Adding Stock

Stock represents the physical piece of material you'll be working with. To add stock to your document:

1. In the **Stock Material** panel in the sidebar, click the **Add Stock** button
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

## Assigning Material

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

## Converting Workpieces to Stock

You can convert any workpiece into a stock item. This is useful when you have an irregular-shaped piece of material and want to use its exact outline as your stock boundary.

To convert a workpiece to stock:

1. Right-click on the workpiece in the canvas or Document panel
2. Select **Convert to Stock** from the context menu
3. The workpiece will be replaced by a new stock item with the same shape and position

The new stock item:

- Uses the workpiece's geometry as its boundary
- Inherits the workpiece's name
- Can be assigned a material like any other stock item

## Auto-Layout

The auto-layout feature helps you efficiently arrange your design elements within stock boundaries:

1. Select the items you want to arrange (or leave nothing selected to arrange all items in the active layer)
2. Click the **Arrange** button in the toolbar and select **Auto Layout (pack workpieces)**
3. Rayforge will automatically arrange the items to optimize material usage

### Auto-Layout Behavior

The auto-layout algorithm arranges items within the visible stock items in your document:

- **If stock items are defined**: Items are arranged within the boundaries of visible stock items
- **If no stock is defined**: Items are arranged across the entire machine workspace

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
5. **Use irregular stock for scrap pieces** - Convert workpieces to stock when using leftover material with custom shapes
6. **Check fit before cutting** - Use the 2D view to verify everything fits on your stock material

## Troubleshooting

### Auto-layout doesn't work as expected

- Make sure at least one stock item is visible
- Make sure items are not grouped (ungroup them first)
- Try reducing the number of items selected at once
- Verify that items fit within the stock boundaries
