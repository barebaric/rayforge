/* --- START OF FILE src/pages/index.js --- */

import React from 'react';
import Layout from '@theme/Layout';
import Link from '@docusaurus/Link';
import styles from './index.module.css';
import Icon from '@mdi/react';
import { mdiRocketLaunch, mdiViewDashboard, mdiFeatureSearch, mdiShareVariant } from '@mdi/js';

function HeroSection() {
  return (
    <section className={styles.hero}>
      <div className={styles.heroInner}>
        
        {/* Left Side: Original Content */}
        <div className={styles.heroContent}>
          <h1 className={styles.heroTitle}>
            The World's <span className={styles.titleGradient}>Most Powerful Open Source</span> Laser Software
          </h1>
          <p className={styles.heroSubtitle}>
            Seamless tools for imaginative minds. Rayforge is the complete creative studio for your laser cutter,
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

function CommunitySection() {
  return (
    <section className={styles.communitySection}>
      <div className={styles.communityInner}>
        <div className={styles.communityDecorations}>
          <div className={styles.communityDot1}></div>
          <div className={styles.communityDot2}></div>
          <div className={styles.communityDot3}></div>
          <div className={styles.communityDot4}></div>
        </div>
        <div className={styles.communityContent}>
          <h2 className={styles.communityTitle}>Made with Rayforge</h2>
          <p className={styles.communitySubtitle}>
            See what creators around the world are making
          </p>
          <a
            href="https://discord.gg/sTHNdTtpQJ"
            className={styles.communityCta}
            target="_blank"
            rel="noopener noreferrer"
          >
            <Icon path={mdiShareVariant} size={0.9} />
            <span>Share your creations</span>
          </a>
        </div>
      </div>
    </section>
  );
}

export default function Home() {
  return (
    <Layout description="Design and create with your laser cutter - the complete creative studio for makers, artists, and crafters">
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
            title="Design Your Own Shapes"
            description="Create custom designs right inside Rayforge. The built-in drawing tools let you sketch, shape, and refine your ideas without needing separate design software."
            features={[
              'Draw lines, circles, curves, and filled shapes',
              'Snap everything into perfect alignment',
              'Set dimensions that update automatically',
            ]}
            image="/assets/screenshot-sketcher.png"
            reverse
          />

          <FeatureSection
            blobClass={styles.blobPurple}
            title="Smart Cutting Paths"
            description="Work faster and waste less material. Rayforge figures out the most efficient way to cut your design, so your laser spends less time moving and more time creating."
            features={[
              "Faster cutting with optimized movement",
              'Save material with automatic layout',
              'Smooth, precise curves',
            ]}
            image="/assets/screenshot-optimizer.png"
          />

          <FeatureSection
            blobClass={styles.blobCyan}
            title="Preview Before You Cut"
            description="Watch your design come to life virtually before committing to the real thing. Spot potential issues early and save yourself from wasted time and materials."
            features={[
              'See exactly how your laser will move',
              'Catch mistakes before you cut',
              'Live preview updates as you design',
            ]}
            image="/screenshots/main-simulation.png"
            reverse
          />

          <FeatureSection
            blobClass={styles.blobOrange}
            title="Material and Workflow Management"
            description="Save your favorite settings and get consistent results across all your projects. Whether you're cutting wood, acrylic, or leather, Rayforge remembers what works best."
            features={[
              'Save settings for different materials',
              'Create recipes you can reuse',
            ]}
            image="/assets/screenshot-recipe.png"
          />
        </div>

        <CommunitySection />

      </main>
    </Layout>
  );
}