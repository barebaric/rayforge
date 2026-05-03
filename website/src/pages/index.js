/* --- START OF FILE src/pages/index.js --- */

import React, { useEffect, useState } from 'react';
import Layout from '@theme/Layout';
import Link from '@docusaurus/Link';
import styles from './index.module.css';
import Icon from '@mdi/react';
import { mdiRocketLaunch, mdiViewDashboard, mdiFeatureSearch, mdiShareVariant } from '@mdi/js';

function detectOs() {
  if (typeof window === 'undefined') {
    return 'linux';
  }

  const userAgent = window.navigator.userAgent.toLowerCase();

  if (userAgent.includes('win')) {
    return 'windows';
  }
  if (
    userAgent.includes('mac') ||
    userAgent.includes('iphone') ||
    userAgent.includes('ipad')
  ) {
    return 'macos';
  }
  if (userAgent.includes('linux')) {
    return 'linux';
  }

  return 'linux';
}

function HeroSection() {
  const [os, setOs] = useState('linux');

  useEffect(() => {
    setOs(detectOs());
  }, []);

  const downloadTo = `/docs/getting-started/installation#${os}`;

  return (
    <section className={styles.hero}>
      <div className={styles.heroInner}>
        
        {/* Left Side: Original Content */}
        <div className={styles.heroContent}>
          <h1 className={styles.heroTitle}>
            Free Open Source <span className={styles.titleGradient}>Laser Cutter Software</span>
          </h1>
          <p className={styles.heroSubtitle}>
            Rayforge is a complete creative studio for your laser cutter,
            combining 2D CAD, CAM, and machine control in one vibrant package.
          </p>
          <div className={styles.heroCtaButtons}>
            <Link
              to={downloadTo}
              className={styles.buttonPrimary}
            >
              Download
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
        <div className={styles.communityPanel}>
          <div className={styles.communityText}>
            <div className={styles.communityBadge}>Community</div>
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
          <div className={styles.communityShowcase} aria-hidden="true">
            <div className={`${styles.showcaseCard} ${styles.showcaseCard1}`}></div>
            <div className={`${styles.showcaseCard} ${styles.showcaseCard2}`}></div>
            <div className={`${styles.showcaseCard} ${styles.showcaseCard3}`}></div>
          </div>
        </div>
      </div>
    </section>
  );
}

export default function Home() {
  return (
    <Layout
      title="Free Open Source Laser Cutter Software"
      description="Rayforge is free open-source laser cutter and engraving software for GRBL-based machines. Design with AI, simulate in 3D, and control your laser — the LightBurn alternative."
    >
      <main className={styles.pageWrapper}>
        
        <HeroSection />

        <QuickLinks />

        <div className={styles.featuresWrapper}>
          <FeatureSection
            blobClass={styles.blobPurple}
            title="AI-Powered Design"
            description="Create designs by simply describing what you want. The AI Workpiece Generator
              turns your text descriptions into laser-ready designs instantly."
            features={[
              'Generate designs from text prompts',
              'No design skills required',
              'Works with any OpenAI-compatible provider',
            ]}
            image="/images/ai-prompt.png"
          />

          <FeatureSection
            blobClass={styles.blobCyan}
            title="Your Creations in 3D"
            description="Seamlessly transition from 2D designs to 3D toolpaths. Rayforge generates G-code for 2-axis, 3-axis, and 4-axis laser cutters and engravers, bridging the gap between flat designs and physical objects."
            compatibilityHeading="Compatible with: Grbl, Smoothieware"
            features={[
              'Simulate your work in full 3D',
              'Cut and engrave round objects',
              'Engrave with step down and rotary axes',
            ]}
            image="/images/screenshot-rotary-closeup.png"
            reverse
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
            image="/images/screenshot-sketcher.png"
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
            image="/images/screenshot-optimizer.png"
            reverse
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
          />

          <FeatureSection
            blobClass={styles.blobOrange}
            title="Material and Workflow Management"
            description="Save your favorite settings and get consistent results across all your projects. Whether you're cutting wood, acrylic, or leather, Rayforge remembers what works best."
            features={[
              'Save settings for different materials',
              'Create recipes you can reuse',
            ]}
            image="/images/screenshot-recipe.png"
            reverse
          />
        </div>

        <CommunitySection />

      </main>
    </Layout>
  );
}
