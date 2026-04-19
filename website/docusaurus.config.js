const webpack = require('webpack');

function rayforgeVersionPlugin() {
  return {
    name: 'rayforge-version-plugin',
    configureWebpack() {
      return {
        plugins: [
          new webpack.DefinePlugin({
            RAYFORGE_VERSION: JSON.stringify(process.env.RAYFORGE_VERSION || '0.0.0'),
            IS_PRERELEASE: JSON.stringify(process.env.IS_PRERELEASE || 'false'),
          }),
        ],
      };
    },
  };
}

module.exports = {
  title: 'Rayforge',
  tagline: 'Free open-source laser cutter software — the LightBurn alternative for GRBL-based laser cutting and engraving',
  url: 'https://rayforge.org',
  baseUrl: '/',
  onBrokenLinks: 'warn',
  favicon: 'images/favicon.png',

  organizationName: 'barebaric',
  projectName: 'rayforge',

  customFields: {
    latestVersion: process.env.RAYFORGE_VERSION || '0.0.0',
    isPrerelease: process.env.IS_PRERELEASE === 'true',
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
          blogDescription: 'News, release notes, tutorials, and tips about Rayforge — free laser cutting and engraving software',
          postsPerPage: 10,
        },
        theme: {
          customCss: require.resolve('./src/css/custom.css'),
        },
        sitemap: {
          changefreq: 'weekly',
          priority: 0.5,
          ignorePatterns: ['/search'],
          filename: 'sitemap.xml',
        },
      },
    ],
  ],

  plugins: [
    rayforgeVersionPlugin,
    [
      require.resolve("@easyops-cn/docusaurus-search-local"),
      {
        hashed: true,
        language: ["en", "de", "es", "fr", "pt", "zh"],
        indexBlog: true,
        indexDocs: true,
        explicitSearchResultPath: true,
      },
    ],
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

  headTags: [
    // JSON-LD: SoftwareApplication schema
    {
      tagName: 'script',
      attributes: {
        type: 'application/ld+json',
      },
      innerHTML: JSON.stringify({
        '@context': 'https://schema.org',
        '@type': 'SoftwareApplication',
        name: 'Rayforge',
        description: 'Free open-source laser cutter software for GRBL-based machines. Design, simulate, and control your laser cutter or engraver with AI-powered tools, 3D preview, and a built-in sketcher.',
        url: 'https://rayforge.org',
        applicationCategory: 'DesignApplication',
        operatingSystem: 'Linux, Windows, macOS',
        offers: {
          '@type': 'Offer',
          price: '0',
          priceCurrency: 'USD',
        },
        author: {
          '@type': 'Organization',
          name: 'barebaric',
          url: 'https://github.com/barebaric',
        },
        screenshot: 'https://rayforge.org/screenshots/main-standard.png',
        featureList: [
          'AI-powered design generation',
          '3D simulation and preview',
          'Parametric 2D sketcher',
          'GRBL and Smoothieware support',
          'Rotary axis support',
          'Material and recipe management',
          'Path optimization',
          'Multi-layer support',
          'Camera calibration',
        ].join(', '),
        license: 'https://github.com/barebaric/rayforge/blob/main/LICENSE',
        programmingLanguage: 'Python',
        installUrl: 'https://rayforge.org/docs/getting-started/installation',
      }),
    },
    // JSON-LD: Organization schema
    {
      tagName: 'script',
      attributes: {
        type: 'application/ld+json',
      },
      innerHTML: JSON.stringify({
        '@context': 'https://schema.org',
        '@type': 'Organization',
        name: 'Rayforge',
        url: 'https://rayforge.org',
        logo: 'https://rayforge.org/images/icon.svg',
        sameAs: [
          'https://github.com/barebaric/rayforge',
          'https://discord.gg/sTHNdTtpQJ',
        ],
      }),
    },
    // JSON-LD: WebSite schema with search action
    {
      tagName: 'script',
      attributes: {
        type: 'application/ld+json',
      },
      innerHTML: JSON.stringify({
        '@context': 'https://schema.org',
        '@type': 'WebSite',
        name: 'Rayforge',
        url: 'https://rayforge.org',
      }),
    },
  ],

  themeConfig: {
    metadata: [
      { name: 'keywords', content: 'laser cutter software, laser engraving software, LightBurn alternative, free laser software, open source laser software, GRBL, laser cutting software, laser control software, G-code sender, Rayforge' },
      { name: 'author', content: 'Rayforge Contributors' },
      { name: 'robots', content: 'index, follow' },
      { property: 'og:type', content: 'website' },
      { property: 'og:site_name', content: 'Rayforge' },
      { property: 'og:image', content: 'https://rayforge.org/images/social.png' },
      { property: 'og:image:width', content: '1200' },
      { property: 'og:image:height', content: '630' },
      { property: 'og:image:alt', content: 'Rayforge - Free Open Source Laser Cutter Software' },
      { name: 'twitter:card', content: 'summary_large_image' },
      { name: 'twitter:title', content: 'Rayforge - Free Open Source Laser Cutter Software' },
      { name: 'twitter:description', content: 'The complete creative studio for your laser cutter. Design, simulate, and control — with AI-powered tools, 3D preview, and a built-in sketcher.' },
      { name: 'twitter:image', content: 'https://rayforge.org/images/social.png' },
      { name: 'twitter:image:alt', content: 'Rayforge - Free Open Source Laser Cutter Software' },
    ],
    navbar: {
      title: 'Rayforge',
      logo: {
        alt: 'Rayforge Logo',
        src: 'images/icon.svg',
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
          to: '/sponsor',
          label: 'Sponsor',
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
