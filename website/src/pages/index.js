/* --- START OF FILE src/pages/index.js --- */

import React, { useEffect, useState } from 'react';
import Layout from '@theme/Layout';
import Link from '@docusaurus/Link';
import styles from './index.module.css';
import Icon from '@mdi/react';
import {
  mdiDownload,
  mdiGithub,
  mdiArrowRight,
  mdiPlayCircleOutline,
  mdiYoutube,
  mdiShareVariant,
  mdiVectorSquare,
  mdiLayersOutline,
  mdiCameraOutline,
  mdiRotate3d,
  mdiBookOpenOutline,
  mdiMapOutline,
} from '@mdi/js';
import { tutorials } from '../data/tutorials';

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

        <div className={styles.heroContent}>
          <p className={styles.kicker}>Design / Prepare / Create</p>
          <h1 className={styles.heroTitle}>
            <span className={styles.heroTitleLine1}>From ideas</span>
            <span className={styles.heroTitleLine2}>to real projects</span>
          </h1>
          <p className={styles.heroSubtitle}>
            Rayforge is the creative suite for your laser cutter. Design,
            prepare and make — all in one free, open-source app.
          </p>
          <div className={styles.heroCtaButtons}>
            <Link to={downloadTo} className={styles.buttonDark}>
              <Icon path={mdiDownload} size={0.85} />
              <span>Download for Free</span>
            </Link>
            <a
              href="https://github.com/barebaric/rayforge"
              className={styles.buttonOutline}
              target="_blank"
              rel="noopener noreferrer"
            >
              <Icon path={mdiGithub} size={0.85} />
              <span>Open Source</span>
            </a>
          </div>
          <a
            href="https://www.youtube.com/watch?v=srKXs2p31VY"
            className={styles.heroVideoLink}
            target="_blank"
            rel="noopener noreferrer"
          >
            <Icon path={mdiPlayCircleOutline} size={0.9} />
            <span>Watch the introduction</span>
            <Icon path={mdiArrowRight} size={0.65} />
          </a>
        </div>

      </div>
    </section>
  );
}

const capabilities = [
  {
    icon: mdiVectorSquare,
    label: '2D CAD Sketcher',
    to: '/docs/features/sketcher',
  },
  {
    icon: mdiLayersOutline,
    label: 'Multi-Layer Jobs',
    to: '/docs/features/multi-layer',
  },
  {
    icon: mdiCameraOutline,
    label: 'Camera Alignment',
    to: '/docs/machine/camera',
  },
  {
    icon: mdiRotate3d,
    label: 'Rotary Support',
    to: '/docs/machine/rotary',
  },
  {
    icon: mdiBookOpenOutline,
    label: 'Material Recipes',
    to: '/docs/application-settings/recipes',
  },
  {
    icon: mdiMapOutline,
    label: 'Path Optimization',
    to: '/docs/features/path-optimization',
  },
];

function CapabilityStrip() {
  return (
    <section className={styles.stripSection}>
      <div className={styles.stripInner}>
        {capabilities.map((cap) => (
          <Link key={cap.label} to={cap.to} className={styles.stripItem}>
            <Icon path={cap.icon} size={1.15} />
            <span>{cap.label}</span>
          </Link>
        ))}
      </div>
    </section>
  );
}

function DesignYourPartsSection() {
  return (
    <section className={styles.partsSection}>
      <div className={styles.partsLayers}>
        <div className={styles.partsLeft} />
        <div className={styles.partsRight} />
      </div>
      <div className={styles.partsInner}>
        <div className={styles.partsContent}>
          <p className={styles.partsKicker}>
            Powerful tools. Endless possibilities.
          </p>
          <h2 className={styles.partsTitle}>Make your own parts</h2>
          <p className={styles.partsText}>
            Sketch, shape and refine custom designs right inside Rayforge. The
            built-in drawing tools let you bring any idea to life — or describe
            what you want and the AI Workpiece Generator designs it for you
            instantly.
          </p>
          <Link to="/docs/features/sketcher" className={styles.partsLink}>
            <span>Learn more</span>
            <Icon path={mdiArrowRight} size={0.65} />
          </Link>
        </div>
      </div>
    </section>
  );
}

function FeatureCardsSection() {
  const cards = [
    {
      title: 'Design',
      subtitle: 'Powerful 2D CAD sketcher with parametric tools.',
      image: '/images/screenshot-sketcher.webp',
    },
    {
      title: 'Prepare',
      subtitle: 'Trace images, optimize toolpaths, and fine-tune every detail.',
      image: '/images/screenshot-optimizer.webp',
    },
    {
      title: 'Create',
      subtitle: 'Run laser and CNC jobs with confidence. Fast. Precise. Reliable.',
      image: '/screenshots/main-3d-bee.webp',
    },
  ];

  return (
    <section className={styles.cardsSection}>
      <div className={styles.cardsTrees}>
        <div className={styles.cardsTreeLeft} />
        <div className={styles.cardsTreeRight} />
      </div>
      <div className={styles.cardsInner}>
        <p className={styles.cardsKicker}>Everything you need. Nothing you don't.</p>
        <div className={styles.cardsGrid}>
          {cards.map((card) => (
            <div key={card.title} className={styles.card}>
              <div className={styles.cardImage}>
                <img src={card.image} alt={card.title} loading="lazy" />
              </div>
              <div className={styles.cardBody}>
                <h3 className={styles.cardTitle}>{card.title}</h3>
                <p className={styles.cardSubtitle}>{card.subtitle}</p>
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

function TutorialSpotlight() {
  return (
    <section className={styles.spotlightSection}>
      <div className={styles.spotlightInner}>
        <div className={styles.spotlightHeader}>
          <p className={styles.kicker}>Community</p>
          <h2 className={styles.spotlightTitle}>Tutorials by real users</h2>
          <p className={styles.spotlightSubtitle}>
            Learn from videos made by actual Rayforge users. Your tutorial
            could be up here next.
          </p>
        </div>

        {tutorials.length > 0 ? (
          <div className={styles.spotlightGrid}>
            {tutorials.map((tutorial) => (
              <a
                key={tutorial.id}
                href={`https://www.youtube.com/watch?v=${tutorial.id}`}
                target="_blank"
                rel="noopener noreferrer"
                className={styles.spotlightCard}
              >
                <div className={styles.spotlightThumb}>
                  <img
                    src={`https://img.youtube.com/vi/${tutorial.id}/hqdefault.jpg`}
                    alt={tutorial.title}
                    loading="lazy"
                  />
                  <span className={styles.spotlightPlay}>
                    <Icon path={mdiPlayCircleOutline} size={1.4} />
                  </span>
                </div>
                <h3 className={styles.spotlightVideoTitle}>{tutorial.title}</h3>
                <span className={styles.spotlightCreator}>
                  {tutorial.creator}
                </span>
              </a>
            ))}
          </div>
        ) : (
          <div className={styles.spotlightEmpty}>
            <div className={styles.spotlightEmptyIcon}>
              <Icon path={mdiYoutube} size={1.6} />
            </div>
            <h3>This spotlight is empty — be its first star.</h3>
            <p>
              Make a Rayforge tutorial, and we will feature it right here on
              the homepage with your name and a link to your channel.
            </p>
            <Link to="/contributing" className={styles.buttonDark}>
              <Icon path={mdiPlayCircleOutline} size={0.85} />
              <span>Make the first tutorial</span>
            </Link>
          </div>
        )}
      </div>
    </section>
  );
}

function CommunitySection() {
  return (
    <section className={styles.communitySection}>
      <div className={styles.communityInner}>
        <p className={styles.kicker}>Showcase</p>
        <h2 className={styles.communityTitle}>Made with Rayforge</h2>
        <p className={styles.communitySubtitle}>
          See what creators around the world are making and share your own
          work.
        </p>
        <a
          href="https://discord.gg/sTHNdTtpQJ"
          className={styles.buttonDark}
          target="_blank"
          rel="noopener noreferrer"
        >
          <Icon path={mdiShareVariant} size={0.85} />
          <span>Share your creations</span>
        </a>
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

        <CapabilityStrip />

        <DesignYourPartsSection />

        <FeatureCardsSection />

        <TutorialSpotlight />

        <CommunitySection />

      </main>
    </Layout>
  );
}
