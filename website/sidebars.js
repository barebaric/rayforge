module.exports = {
  tutorialSidebar: [
    {
      type: 'category',
      label: 'Getting Started',
      collapsed: false,
      items: [
        'getting-started/installation',
        'getting-started/first-time-setup',
        'getting-started/quick-start',
      ],
    },
    {
      type: 'category',
      label: 'Application Settings',
      items: [
        'ui/settings',
        'application-settings/machines',
        'application-settings/materials',
        'application-settings/recipes',
        'application-settings/ai-provider',
        'application-settings/addons',
      ],
    },
    {
      type: 'category',
      label: 'Machine Setup',
      items: [
        'machine/general',
        'machine/hardware',
        'machine/advanced',
        'machine/gcode',
        'machine/hooks-macros',
        'machine/device',
        'machine/laser',
        'machine/rotary',
        'machine/nogo-zones',
        'machine/camera',
        'machine/maintenance',
      ],
    },
    {
      type: 'category',
      label: 'User Interface',
      items: [
        'ui/main-window',
        'ui/canvas-tools',
        'ui/3d-preview',
        'ui/bottom-panel',
      ],
    },
        {
          type: 'category',
          label: 'Features',
          items: [
            {
              type: 'category',
              label: 'Operations',
              items: [
                'features/operations/contour',
                'features/operations/engrave',
                'features/operations/shrink-wrap',
                'features/operations/frame-outline',
                'features/operations/material-test-grid',
              ],
            },
            {
              type: 'category',
              label: 'Post-Processing',
              items: [
                'features/smooth',
                'features/multi-pass',
                'features/path-optimization',
                'features/crop-to-stock',
                'features/overscan',
                'features/holding-tabs',
                'features/merge-lines',
                'features/lead-in-out',
              ],
            },
            'features/sketcher',
            'features/multi-layer',
            'features/stock-handling',
            'features/simulation-mode',
            'features/framing-your-job',
            'features/workpiece-positioning',
            'features/kerf',
          ],
        },
    {
      type: 'category',
      label: 'Addons',
      items: [
        'addons/ai-workpiece-generator',
      ],
    },
    {
      type: 'category',
      label: 'Files',
      items: [
        'files/importing',
        'files/formats',
        'files/exporting',
      ],
    },
    {
      type: 'category',
      label: 'General Info',
      items: [
        'general-info/laser-safety',
        'general-info/coordinate-systems',
        'general-info/gcode-basics',
        'general-info/usage-tracking',
      ],
    },
    {
      type: 'category',
      label: 'Troubleshooting',
      items: [
        'troubleshooting/connection',
        'troubleshooting/debug',
        'troubleshooting/snap-permissions',
      ],
    },
    {
      type: 'category',
      label: 'Reference',
      items: [
        'reference/shortcuts',
        'reference/gcode-dialects',
        'reference/firmware',
      ],
    },
  ],
  developerSidebar: [
    {
      type: 'category',
      label: 'Getting Started',
      collapsed: false,
      items: [
        'developer/getting-started/index',
        'developer/getting-started/setup',
        'developer/getting-started/submitting-changes',
      ],
    },
    {
      type: 'category',
      label: 'Architecture',
      items: [
        'developer/architecture',
        'developer/docmodel',
        'developer/importer',
        'developer/pipeline',
        'developer/tasker',
        'developer/driver',
      ],
    },
    {
      type: 'category',
      label: 'Addon Development',
      items: [
        'developer/addon-overview',
        'developer/addon-manifest',
        'developer/addon-hooks',
        'developer/addon-registries',
      ],
    },
  ],
};
