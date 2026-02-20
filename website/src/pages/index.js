/* --- START OF FILE src/pages/index.js --- */

import React from 'react';
import Layout from '@theme/Layout';
import Link from '@docusaurus/Link';
import styles from './index.module.css';
import Icon from '@mdi/react';
import { mdiRocketLaunch, mdiViewDashboard, mdiFeatureSearch } from '@mdi/js';

function HeroSection() {
  return (
    <section className={styles.hero}>
      <div className={styles.heroInner}>
        
        {/* Left Side: Original Content */}
        <div className={styles.heroContent}>
          <h1 className={styles.heroTitle}>
            The World's Most <br />
            <span className={styles.titleGradient}>Powerful Open Source </span>
            Laser Software
          </h1>
          <p className={styles.heroSubtitle}>
            Seamless tools for imaginative minds. Rayforge is the open-source power tool for your laser cutter,
            combining 2D CAD, CAM, and machine control in one vibrant package.
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

        {/* Right Side: Visual Cluster (Screenshot + Video) */}
        <div className={styles.heroVisuals}>
          <div className={styles.visualBlob}></div>
          
          {/* Main Back Layer: Screenshot */}
          <div className={styles.screenshotLayer}>
            <img
              src="/screenshots/main-standard.png"
              alt="Rayforge application screenshot"
              className={styles.heroImage}
            />
          </div>

          {/* Floating Front Layer: Video */}
          <div className={styles.videoLayer}>
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
          <div className={`${styles.card} ${styles.cardCyan}`}>
            <div className={styles.cardIcon}>
              <Icon path={mdiRocketLaunch} size={1.5} />
            </div>
            <h3>Getting Started</h3>
            <p>Install and configure Rayforge.</p>
          </div>
        </Link>
        
        {/* Purple Theme Card */}
        <Link to="/docs/ui/main-window" className={styles.cardLink}>
          <div className={`${styles.card} ${styles.cardPurple}`}>
            <div className={styles.cardIcon}>
              <Icon path={mdiViewDashboard} size={1.5} />
            </div>
            <h3>User Interface</h3>
            <p>Explore the tools and layout.</p>
          </div>
        </Link>
        
        {/* Orange Theme Card */}
        <Link to="/docs/features/sketcher" className={styles.cardLink}>
          <div className={`${styles.card} ${styles.cardOrange}`}>
            <div className={styles.cardIcon}>
              <Icon path={mdiFeatureSearch} size={1.5} />
            </div>
            <h3>Features</h3>
            <p>Discover more capabilities.</p>
          </div>
        </Link>
      </div>
    </div>
  );
}

function FeatureSection({ 
  title, 
  description, 
  features, 
  image, 
  reverse, 
  compatibilityHeading,
  blobClass 
}) {
  return (
    <section className={styles.featureSection}>
      <div className={`${styles.featureSectionInner} ${reverse ? styles.reverse : ''}`}>
        
        <div className={styles.featureVisual}>
          <div className={`${styles.blobShape} ${blobClass}`}></div>
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

        <QuickLinks />

        <div className={styles.featuresWrapper}>
          <FeatureSection
            blobClass={styles.blobCyan}
            title="Your Creations in 3D"
            description="Seamlessly transition from 2D designs to 3D toolpaths. Rayforge generates G-code for both 2-axis and 3-axis laser cutters and engravers, bridging the gap between flat designs and physical objects."
            compatibilityHeading="Compatible with: Grbl, Smoothieware"
            features={['3D Visualization', 'Toolpath Generation', 'Axis Control']}
            image="/assets/screenshot-3d-closeup.png"
          />

          <FeatureSection
            blobClass={styles.blobOrange}
            title="Parametric 2D Sketcher"
            description="Create precise, constraint-based 2D designs directly within Rayforge. The parametric sketcher provides a complete set of tools for creating geometric shapes and applying parametric constraints."
            features={[
              'Create lines, circles, arcs, and fills',
              'Apply geometric constraints: coincident, vertical, horizontal...',
              'Parametric expressions for calculated dimensions',
            ]}
            image="/assets/screenshot-sketcher.png"
            reverse
          />

          <FeatureSection
            blobClass={styles.blobPurple}
            title="Advanced Path Optimization"
            description="Save time and material with intelligent pathfinding algorithms and auto-layout functionality. Rayforge automatically optimizes the cutting order to minimize travel time."
            features={[
              "Don't waste time when cutting and engraving",
              'Save material with automatic layout and nesting',
              'Native arc support with G2/G3 commands',
            ]}
            image="/assets/screenshot-optimizer.png"
          />

          <FeatureSection
            blobClass={styles.blobCyan}
            title="Powerful Built-in Simulator"
            description="Validate your G-code and simulate the cutting process before you start. Our built-in simulator helps you catch potential issues early, saving you time and materials."
            features={[
              'G-code preview',
              'Visual simulation of cutting paths',
              'Live 2D preview of all operations while you edit',
            ]}
            image="/screenshots/main-simulation.png"
            reverse
          />

          <FeatureSection
            blobClass={styles.blobOrange}
            title="Material and Workflow Management"
            description="Streamline your laser cutting workflow with comprehensive material, stock, and recipe management tools designed to improve consistency and efficiency."
            features={[
              'Organize materials by type and properties',
              'Define physical stock dimensions',
              'Create reusable recipes for consistent results',
            ]}
            image="/assets/screenshot-recipe.png"
          />
        </div>

      </main>
    </Layout>
  );
}