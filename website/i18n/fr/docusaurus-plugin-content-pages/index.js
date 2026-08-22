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
          <p className={styles.kicker}>Concevoir / Préparer / Créer</p>
          <h1 className={styles.heroTitle}>
            <span className={styles.heroTitleLine1}>Des idées</span>
            <span className={styles.heroTitleLine2}>à de vrais projets</span>
          </h1>
          <p className={styles.heroSubtitle}>
            Rayforge est la suite créative pour votre découpeuse laser.
            Concevez, préparez et créez — le tout dans une application libre
            et gratuite.
          </p>
          <div className={styles.heroCtaButtons}>
            <Link to={downloadTo} className={styles.buttonDark}>
              <Icon path={mdiDownload} size={0.85} />
              <span>Télécharger gratuitement</span>
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
            <span>Voir l'introduction</span>
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
    label: 'Esquisse 2D CAD',
    to: '/docs/features/sketcher',
  },
  {
    icon: mdiLayersOutline,
    label: 'Jobs multi-couches',
    to: '/docs/features/multi-layer',
  },
  {
    icon: mdiCameraOutline,
    label: 'Alignement par caméra',
    to: '/docs/machine/camera',
  },
  {
    icon: mdiRotate3d,
    label: 'Support rotatif',
    to: '/docs/machine/rotary',
  },
  {
    icon: mdiBookOpenOutline,
    label: 'Recettes de matériaux',
    to: '/docs/application-settings/recipes',
  },
  {
    icon: mdiMapOutline,
    label: 'Optimisation des trajets',
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
            Des outils puissants. Des possibilités infinies.
          </p>
          <h2 className={styles.partsTitle}>Créez vos propres pièces</h2>
          <p className={styles.partsText}>
            Esquissez, façonnez et perfectionnez vos designs directement dans
            Rayforge. Les outils de dessin intégrés donnent vie à toutes vos
            idées — ou décrivez simplement ce que vous voulez et le générateur
            de pièces par IA le conçoit instantanément.
          </p>
          <Link to="/docs/features/sketcher" className={styles.partsLink}>
            <span>En savoir plus</span>
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
      title: 'Concevoir',
      subtitle: "Esquisseur 2D CAD puissant avec outils paramétriques.",
      image: '/images/screenshot-sketcher.webp',
    },
    {
      title: 'Préparer',
      subtitle:
        "Tracez des images, optimisez les trajectoires et peaufinez chaque détail.",
      image: '/images/screenshot-optimizer.webp',
    },
    {
      title: 'Créer',
      subtitle:
        "Exécutez vos travaux laser et CNC en toute confiance. Rapide. Précis. Fiable.",
      image: '/screenshots/main-3d-bee.png',
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
          Tout ce qu'il faut. Rien de superflu.
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
          <p className={styles.kicker}>Communauté</p>
          <h2 className={styles.spotlightTitle}>
            Des tutoriels d'utilisateurs réels
          </h2>
          <p className={styles.spotlightSubtitle}>
            Apprenez grâce à des vidéos créées par de vrais utilisateurs de
            Rayforge. Votre tutoriel pourrait être le prochain affiché ici.
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
            <h3>Cet espace est vide — soyez sa première étoile.</h3>
            <p>
              Créez un tutoriel Rayforge et nous le mettrons en avant ici même,
              sur la page d'accueil, avec votre nom et un lien vers votre
              chaîne.
            </p>
            <Link to="/contributing" className={styles.buttonDark}>
              <Icon path={mdiPlayCircleOutline} size={0.85} />
              <span>Créer le premier tutoriel</span>
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
        <p className={styles.kicker}>Galerie</p>
        <h2 className={styles.communityTitle}>Réalisé avec Rayforge</h2>
        <p className={styles.communitySubtitle}>
          Découvrez ce que des créateurs du monde entier réalisent et
          partagez votre propre travail.
        </p>
        <a
          href="https://discord.gg/sTHNdTtpQJ"
          className={styles.buttonDark}
          target="_blank"
          rel="noopener noreferrer"
        >
          <Icon path={mdiShareVariant} size={0.85} />
          <span>Partagez vos créations</span>
        </a>
      </div>
    </section>
  );
}

export default function Home() {
  return (
    <Layout
      title="Logiciel libre et gratuit de découpe laser"
      description="Rayforge est un logiciel libre et gratuit de découpe et de gravure laser pour les machines à base de GRBL. Concevez avec l'IA, simulez en 3D et contrôlez votre laser — l'alternative à LightBurn."
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
