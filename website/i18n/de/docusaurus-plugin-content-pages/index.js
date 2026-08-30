import React, { useEffect, useState } from 'react';
import Layout from '@theme/Layout';
import Link from '@docusaurus/Link';
import styles from '@site/src/pages/index.module.css';
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
import { tutorials } from '@site/src/data/tutorials';

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
          <p className={styles.kicker}>Entwerfen / Vorbereiten / Erstellen</p>
          <h1 className={styles.heroTitle}>
            <span className={styles.heroTitleLine1}>Von der Idee</span>
            <span className={styles.heroTitleLine2}>zum echten Projekt</span>
          </h1>
          <p className={styles.heroSubtitle}>
            Rayforge ist die kreative Suite für deinen Lasercutter. Entwerfen,
            vorbereiten und erstellen – alles in einer kostenlosen
            Open-Source-App.
          </p>
          <div className={styles.heroCtaButtons}>
            <Link to={downloadTo} className={styles.buttonDark}>
              <Icon path={mdiDownload} size={0.85} />
              <span>Kostenlos herunterladen</span>
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
            <span>Die Einführung ansehen</span>
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
    label: '2D-CAD-Sketcher',
    to: '/docs/features/sketcher',
  },
  {
    icon: mdiLayersOutline,
    label: 'Mehrschicht-Aufträge',
    to: '/docs/features/multi-layer',
  },
  {
    icon: mdiCameraOutline,
    label: 'Kamera-Ausrichtung',
    to: '/docs/machine/camera',
  },
  {
    icon: mdiRotate3d,
    label: 'Rotationsachsen',
    to: '/docs/machine/rotary',
  },
  {
    icon: mdiBookOpenOutline,
    label: 'Material-Rezepte',
    to: '/docs/application-settings/recipes',
  },
  {
    icon: mdiMapOutline,
    label: 'Pfadoptimierung',
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
            Starke Werkzeuge. Grenzenlose Möglichkeiten.
          </p>
          <h2 className={styles.partsTitle}>Erstelle deine eigenen Teile</h2>
          <p className={styles.partsText}>
            Skizziere, forme und verfeinere eigene Designs direkt in Rayforge.
            Mit den integrierten Zeichenwerkzeugen bringst du jede Idee zum
            Leben – oder beschreibe einfach, was du möchtest, und der
            KI-Werkstückgenerator entwirft es sofort für dich.
          </p>
          <Link to="/docs/features/sketcher" className={styles.partsLink}>
            <span>Mehr erfahren</span>
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
      title: 'Entwerfen',
      subtitle: 'Leistungsstarker 2D-CAD-Sketcher mit parametrischen Werkzeugen.',
      image: '/images/screenshot-sketcher.webp',
    },
    {
      title: 'Vorbereiten',
      subtitle:
        'Bilder nachzeichnen, Werkzeugpfade optimieren und jedes Detail feinjustieren.',
      image: '/images/screenshot-optimizer.webp',
    },
    {
      title: 'Erstellen',
      subtitle:
        'Führe Laser- und CNC-Aufträge souverän aus. Schnell. Präzise. Zuverlässig.',
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
        <p className={styles.cardsKicker}>
          Alles, was du brauchst. Nichts Überflüssiges.
        </p>
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
          <h2 className={styles.spotlightTitle}>
            Tutorials von echten Nutzern
          </h2>
          <p className={styles.spotlightSubtitle}>
            Lerne aus Videos von echten-Rayforge-Nutzern. Dein Tutorial könnte
            als Nächstes hier erscheinen.
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
            <h3>Dieses Spotlight ist leer – werde sein erster Star.</h3>
            <p>
              Erstelle ein Rayforge-Tutorial, und wir zeigen es direkt hier auf
              der Startseite – mit deinem Namen und einem Link zu deinem Kanal.
            </p>
            <Link to="/contributing" className={styles.buttonDark}>
              <Icon path={mdiPlayCircleOutline} size={0.85} />
              <span>Erstelle das erste Tutorial</span>
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
        <h2 className={styles.communityTitle}>Mit Rayforge erstellt</h2>
        <p className={styles.communitySubtitle}>
          Sieh, was Creator weltweit erstellen, und teile deine eigenen Werke.
        </p>
        <a
          href="https://discord.gg/sTHNdTtpQJ"
          className={styles.buttonDark}
          target="_blank"
          rel="noopener noreferrer"
        >
          <Icon path={mdiShareVariant} size={0.85} />
          <span>Teile deine Kreationen</span>
        </a>
      </div>
    </section>
  );
}

export default function Home() {
  return (
    <Layout
      title="Kostenlose Open-Source-Lasercutter-Software"
      description="Rayforge ist eine kostenlose Open-Source-Software für Laserschneiden und -gravur für GRBL-basierte Maschinen. Entwirf mit KI, simuliere in 3D und steuere deinen Laser – die LightBurn-Alternative."
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
