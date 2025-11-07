# Recipes and Settings

Rayforge provides a powerful recipe system that allows you to create, manage, and apply
consistent settings across your laser cutting projects. This guide covers the complete
user journey from creating recipes in the general settings to applying them to operations
and managing settings at the step level.

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

![Recipe manager](../images/recipe-manager.png)

#### 2. Create New Recipe

Click "Add New Recipe". Fill in the basic information:

- **Name**: Descriptive name (e.g., "3mm Plywood Cut")
- **Description**: Optional detailed description

![Recipe manager](../images/recipe-editor.png)

#### 3. Define Applicability Criteria

- **Task Type**: Select the operation type (Cut, Engrave, etc.)
- **Machine**: Choose a specific machine or leave as "Any Machine"
- **Material**: Select a material type or leave open for any material
- **Thickness Range**: Set minimum and maximum thickness values

#### 4. Configure Settings

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

- [Material Libraries](material-libraries.md) - Managing material properties
- [Stock Handling](stock-handling.md) - Working with stock materials
- [Machine Setup](../machine/index.md) - Configuring machines and laser heads
- [Operations Overview](operations/index.md) - Understanding different operation types
