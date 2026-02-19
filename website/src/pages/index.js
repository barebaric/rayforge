import React from 'react';
import Layout from '@theme/Layout';
import Link from '@docusaurus/Link';
import styles from './index.module.css';

function HeroSection() {
  return (
    <section className={styles.hero}>
      <div className={styles.heroTop}>
        <div className={styles.heroContent}>
          <h1 className={styles.heroTitle}>
            The World's Most <br />
            <span className={styles.titleGradient}>Powerful Open Source</span> <br />
            Laser Software
          </h1>
          <p className={styles.heroSubtitle}>
            Precision laser cutting and engraving for limitless creations.
            2D CAD. Completely free and open source.
          </p>
          <div className={styles.heroCtaButtons}>
            <Link
              to="/docs/getting-started/installation"
              className={styles.buttonPrimary}
            >
              Get Started
            </Link>
            <a
              href="https://github.com/barebaric/rayforge"
              className={styles.buttonSecondary}
              target="_blank"
              rel="noopener noreferrer"
            >
              View on GitHub
            </a>
          </div>
        </div>
        <div className={styles.heroImageContainer}>
          <img
            src="/assets/screenshot.png"
            alt="Rayforge application screenshot"
            className={styles.heroImage}
          />
        </div>
      </div>
    </section>
  );
}

function QuickLinks() {
  return (
    <div className={styles.linksContainer}>
      <div className={styles.cardGrid}>
        {/* Cyan Theme Card */}
        <Link to="/docs/getting-started/installation" className={styles.cardLink}>
          <div className={`${styles.card} ${styles.cardThemeCyan}`}>
            <div className={styles.cardIcon}>
              <RocketIcon />
            </div>
            <h3>Getting Started</h3>
            <p>Install and configure Rayforge.</p>
          </div>
        </Link>
        
        {/* Purple Theme Card */}
        <Link to="/docs/ui/main-window" className={styles.cardLink}>
          <div className={`${styles.card} ${styles.cardThemePurple}`}>
            <div className={styles.cardIcon}>
              <DashboardIcon />
            </div>
            <h3>User Interface</h3>
            <p>Explore the tools and layout.</p>
          </div>
        </Link>
        
        {/* Orange Theme Card */}
        <Link to="/docs/features/sketcher" className={styles.cardLink}>
          <div className={`${styles.card} ${styles.cardThemeOrange}`}>
            <div className={styles.cardIcon}>
              <FeatureSearchIcon />
            </div>
            <h3>Features</h3>
            <p>Discover more capabilities.</p>
          </div>
        </Link>
      </div>
    </div>
  );
}

function RocketIcon() {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="32" height="32">
      <path fill="currentColor" d="M9.19 6.35c-2.04 2.29-3.44 5.58-3.57 5.89L2 10.69l4.05-4.05c.47-.47 1.15-.68 1.81-.55l1.33.26zM11.17 17s3.74-1.55 5.89-3.7c5.4-5.4 4.5-9.62 4.21-10.57-.95-.3-5.17-1.19-10.57 4.21C8.55 9.09 7 12.83 7 12.83L11.17 17zm6.48-2.19c-2.29 2.04-5.58 3.44-5.89 3.57L13.31 22l4.05-4.05c.47-.47.68-1.15.55-1.81l-.26-1.33zM9 18c0 .83-.34 1.58-.88 2.12C6.94 21.3 2 22 2 22s.7-4.94 1.88-6.12A2.996 2.996 0 0 1 9 18zm3-5a1 1 0 1 1 0-2 1 1 0 0 1 0 2z"/>
    </svg>
  );
}

function DashboardIcon() {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="32" height="32">
      <path fill="currentColor" d="M3 13h8V3H3v10zm0 8h8v-6H3v6zm10 0h8V11h-8v10zm0-18v6h8V3h-8z"/>
    </svg>
  );
}

function FeatureSearchIcon() {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="32" height="32">
      <path fill="currentColor" d="M20 20.94L22.14 18.8L19.7 16.36C20.49 15.13 21 13.63 21 12c0-4.97-4.03-9-9-9s-9 4.03-9 9 4.03 9 9 9c1.63 0 3.13-.51 4.36-1.3L18.79 22.14 20 20.94zM12 19c-3.86 0-7-3.14-7-7s3.14-7 7-7 7 3.14 7 7-3.14 7-7 7z"/>
      <circle fill="currentColor" cx="12" cy="12" r="3"/>
    </svg>
  );
}

function FeatureSection({ 
  title, 
  description, 
  features, 
  image, 
  reverse, 
  compatibilityHeading,
  bgColorClass, // New prop for background color
  blobColorClass // New prop for image blob decoration
}) {
  return (
    <section className={`${styles.featureSection} ${bgColorClass}`}>
      <div className={`${styles.featureSectionInner} ${reverse ? styles.reverse : ''}`}>
        
        <div className={`${styles.featureVisual} ${blobColorClass}`}>
          <img src={image} alt={title} />
        </div>

        <div className={styles.featureContent}>
          <h2>{title}</h2>
          <p>{description}</p>
          {compatibilityHeading && <p className={styles.compatibilityHeading}>{compatibilityHeading}</p>}
          {features && (
            <ul>
              {features.map((f, i) => <li key={i}>{f}</li>)}
            </ul>
          )}
        </div>
        
      </div>
    </section>
  );
}

export default function Home() {
  return (
    <Layout description="Modern G-code sender and control software for GRBL-based laser cutters">
      <main className={styles.pageWrapper}>
        
        <HeroSection />

        <div className={styles.videoSection}>
          <div className={styles.videoWrapper}>
            <iframe
              src="https://www.youtube.com/embed/srKXs2p31VY"
              title="Rayforge Introduction"
              frameBorder="0"
              allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
              allowFullScreen>
            </iframe>
          </div>
        </div>

        <QuickLinks />

        {/* Feature Sections with Alternating Backgrounds and Blob Colors */}
        <div className={styles.featuresWrapper}>
          
          <FeatureSection
            bgColorClass={styles.bgTintCyan} // Cyan background tint
            blobColorClass={styles.blobCyan}
            title="Your Creations in 3D"
            description="Seamlessly transition from 2D designs to 3D toolpaths. Rayforge generates G-code for both 2-axis and 3-axis laser cutters and engravers, bridging the gap between flat designs and physical objects."
            compatibilityHeading="Compatible with:"
            features={['Grbl', 'Smoothieware']}
            image="/assets/screenshot-3d-closeup.png"
          />

          <FeatureSection
            bgColorClass={styles.bgWhite} // Clean White
            blobColorClass={styles.blobOrange}
            title="Parametric 2D Sketcher"
            description="Create precise, constraint-based 2D designs directly within Rayforge. The parametric sketcher provides a complete set of tools for creating geometric shapes and applying parametric constraints."
            features={[
              'Create lines, circles, arcs, and fills',
              'Apply geometric constraints: coincident, vertical, horizontal...',
              'Set dimensional constraints: distance, diameter, radius...',
              'Parametric expressions for calculated dimensions',
              'Construction mode for helper geometry',
            ]}
            image="/assets/screenshot-sketcher.png"
            reverse
          />

          <FeatureSection
            bgColorClass={styles.bgTintOrange} // Orange background tint
            blobColorClass={styles.blobPurple}
            title="Advanced Path Optimization"
            description="Save time and material with intelligent pathfinding algorithms and auto-layout functionality. Rayforge automatically optimizes the cutting order to minimize travel time."
            features={[
              "Don't waste time when cutting and engraving",
              'Save material with automatic layout and nesting for large projects',
              'Native arc support with G2/G3 commands',
            ]}
            image="/assets/screenshot-optimizer.png"
          />

          <FeatureSection
            bgColorClass={styles.bgWhite} // Clean White
            blobColorClass={styles.blobCyan}
            title="Powerful Built-in Simulator"
            description="Validate your G-code and simulate the cutting process before you start. Our built-in simulator helps you catch potential issues early, saving you time and materials."
            features={[
              'G-code preview',
              'Visual simulation of cutting paths',
              'Live 2D preview of all operations while you edit',
            ]}
            image="/assets/screenshot-simulation.png"
            reverse
          />

          <FeatureSection
            bgColorClass={styles.bgTintPurple} // Purple background tint
            blobColorClass={styles.blobOrange}
            title="Material and Workflow Management"
            description="Streamline your laser cutting workflow with comprehensive material, stock, and recipe management tools designed to improve consistency and efficiency."
            features={[
              'Organize materials by type and properties with custom libraries',
              'Define physical stock dimensions and optimize material usage',
              'Create reusable recipes for consistent results',
              'Auto-layout designs within stock boundaries',
            ]}
            image="/assets/screenshot-recipe.png"
          />
        </div>

      </main>
    </Layout>
  );
}