# Recipes and Settings

![Recipes Settings](/images/application-recipes.png)

Rayforge provides a powerful recipe system that allows you to create,
manage, and apply consistent settings across your laser cutting projects.
This guide covers the complete user journey from creating recipes in the
general settings to applying them to operations and managing settings at
the step level.

## Overview

The recipe system consists of three main components:

1. **Recipe Management**: Create and manage reusable settings presets
2. **Stock Material Management**: Define material properties and thickness
3. **Step Settings**: Apply and fine-tune settings for individual operations

## Recipe Management

### Creating Recipes

Recipes are named presets that contain all the settings needed for specific operations.
You can create recipes through the main settings interface:

#### 1. Access Recipe Manager

Menu: Edit → Preferences → Recipes

#### 2. Create New Recipe

Click "Add New Recipe" to open the recipe editor dialog.

**General Tab** - Set the recipe name and description:

![Recipe Editor - General Tab](/images/recipe-editor-general.png)

Fill in the basic information:

- **Name**: Descriptive name (e.g., "3mm Plywood Cut")
- **Description**: Optional detailed description

#### 3. Define Applicability Criteria

**Applicability Tab** - Define when this recipe should be suggested:

![Recipe Editor - Applicability Tab](/images/recipe-editor-applicability.png)

- **Task Type**: Select the operation type (Cut, Engrave, etc.)
- **Machine**: Choose a specific machine or leave as "Any Machine"
- **Material**: Select a material type or leave open for any material
- **Thickness Range**: Set minimum and maximum thickness values

#### 4. Configure Settings

**Settings Tab** - Adjust power, speed, and other parameters:

![Recipe Editor - Settings Tab](/images/recipe-editor-settings.png)

- Adjust power, speed, and other parameters
- Settings automatically adapt based on the selected task type

### Recipe Matching System

Rayforge automatically suggests the most appropriate recipes based on:

- **Machine compatibility**: Recipes can be machine-specific
- **Material matching**: Recipes can target specific materials
- **Thickness ranges**: Recipes apply within defined thickness limits
- **Capability matching**: Recipes are tied to specific operation types

The system uses a specificity scoring algorithm to prioritize the most relevant recipes:

1. Machine-specific recipes rank higher than generic ones
2. Laser head-specific recipes rank higher
3. Material-specific recipes rank higher
4. Thickness-specific recipes rank higher

---

**Related Topics**:

- [Materials](materials) - Managing material properties
- [Stock Handling](../features/stock-handling) - Working with stock materials
- [Machine Setup](../machine/general) - Configuring machines and laser heads
- [Operations Overview](../features/operations/contour) - Understanding different operation types
