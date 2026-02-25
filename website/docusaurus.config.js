const webpack = require('webpack');

function rayforgeVersionPlugin() {
  return {
    name: 'rayforge-version-plugin',
    configureWebpack() {
      return {
        plugins: [
          new webpack.DefinePlugin({
            RAYFORGE_VERSION: JSON.stringify(process.env.RAYFORGE_VERSION || '0.0.0'),
          }),
        ],
      };
    },
  };
}

module.exports = {
  title: 'Rayforge',
  tagline: 'Modern G-code sender and control software for GRBL-based laser cutters and engravers',
  url: 'https://rayforge.org',
  baseUrl: '/',
  onBrokenLinks: 'warn',
  favicon: 'assets/favicon.png',

  organizationName: 'barebaric',
  projectName: 'rayforge',

  customFields: {
    latestVersion: process.env.RAYFORGE_VERSION || '0.0.0',
  },

  i18n: {
    defaultLocale: 'en',
    locales: ['en', 'pt-BR', 'es', 'fr', 'de', 'zh-CN', 'uk'],
  },

  presets: [
    [
      '@docusaurus/preset-classic',
      {
        docs: {
          path: 'docs',
          routeBasePath: 'docs',
          sidebarPath: require.resolve('./sidebars.js'),
          editUrl: 'https://github.com/barebaric/rayforge/tree/main/website/docs/',
        },
        blog: {
          path: 'blog',
          routeBasePath: 'blog',
          blogTitle: 'Rayforge Blog',
          blogDescription: 'News, updates, tutorials, and tips about Rayforge',
          postsPerPage: 10,
        },
        theme: {
          customCss: require.resolve('./src/css/custom.css'),
        },
      },
    ],
  ],

  plugins: [
    rayforgeVersionPlugin,
    [
      '@docusaurus/theme-mermaid',
      {
        mermaid: {
          theme: 'base',
          themeVariables: {
            primaryColor: '#fff3e0',
            primaryTextColor: '#e65100',
            primaryBorderColor: '#ffb74d',
            lineColor: '#5f27cd',
            secondaryColor: '#e1f5fe',
            tertiaryColor: '#f3e5f5',
            background: '#ffffff',
            mainBkg: '#fff3e0',
            secondBkg: '#fff3e0',
            nodeBkg: '#fff3e0',
            nodeBorder: '#ffb74d',
            clusterBkg: '#fff3e0',
            clusterBorder: '#ffb74d',
            titleColor: '#e65100',
          },
        },
      },
    ],
  ],

  markdown: {
    mermaid: true,
    format: 'detect',
    hooks: {
      onBrokenMarkdownLinks: 'warn',
    },
  },

  scripts: [
    {
      src: 'https://analytics.barebaric.com/script.js',
      'data-website-id': '4493e023-327b-427c-980f-54a49129c732',
      defer: true,
    },
  ],

  themeConfig: {
    navbar: {
      title: 'Rayforge',
      logo: {
        alt: 'Rayforge Logo',
        src: 'assets/icon.svg',
      },
      items: [
        {
          type: 'doc',
          docId: 'getting-started/installation',
          position: 'left',
          label: 'Documentation',
        },
        {
          type: 'docSidebar',
          sidebarId: 'developerSidebar',
          position: 'left',
          label: 'Developer',
        },
        {
          to: '/contributing',
          label: 'Contributing',
          position: 'left',
        },
        {
          to: '/blog',
          label: 'Blog',
          position: 'left',
        },
        {
          type: 'localeDropdown',
          position: 'right',
        },
        {
          href: 'https://github.com/barebaric/rayforge',
          label: 'GitHub',
          position: 'right',
        },
        {
          href: 'https://discord.gg/sTHNdTtpQJ',
          label: 'Discord',
          position: 'right',
        },
      ],
    },
    footer: {
      style: 'dark',
      copyright: `Copyright © ${new Date().getFullYear()} Rayforge Contributors`,
    },
  },
};
