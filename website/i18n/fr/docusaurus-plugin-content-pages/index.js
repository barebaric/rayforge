import React from 'react';
import Layout from '@theme/Layout';
import Link from '@docusaurus/Link';
import styles from '@site/src/pages/index.module.css';
import Icon from '@mdi/react';
import { mdiRocketLaunch, mdiViewDashboard, mdiFeatureSearch } from '@mdi/js';

function HeroSection() {
  return (
    <section className={styles.hero}>
      <div className={styles.heroInner}>
        
        {/* Left Side: Original Content */}
        <div className={styles.heroContent}>
          <h1 className={styles.heroTitle}>
            Le Logiciel Laser Open Source <br />
            <span className={styles.titleGradient}>le Plus Puissant </span>
            au Monde
          </h1>
          <p className={styles.heroSubtitle}>
            Des outils fluides pour les esprits imaginatifs. Rayforge est l'outil open source puissant pour votre découpeuse laser,
            combinant CAO 2D, FAO et contrôle de machine dans un package vibrant.
          </p>
          <div className={styles.heroCtaButtons}>
            <Link
              to="/docs/getting-started/installation"
              className={styles.buttonPrimary}
            >
              Commencer
            </Link>
            <a
              href="https://github.com/barebaric/rayforge"
              className={styles.buttonSecondary}
              target="_blank"
              rel="noopener noreferrer"
            >
              Voir sur GitHub
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
              alt="Capture d'écran de l'application Rayforge"
              className={styles.heroImage}
            />
          </div>

          {/* Floating Front Layer: Video */}
          <div className={styles.videoLayer}>
            <div className={styles.videoWrapper}>
              <iframe
                src="https://www.youtube.com/embed/srKXs2p31VY"
                title="Introduction à Rayforge"
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
            <h3>Premiers Pas</h3>
            <p>Installer et configurer Rayforge.</p>
          </div>
        </Link>
        
        {/* Purple Theme Card */}
        <Link to="/docs/ui/main-window" className={styles.cardLink}>
          <div className={`${styles.card} ${styles.cardPurple}`}>
            <div className={styles.cardIcon}>
              <Icon path={mdiViewDashboard} size={1.5} />
            </div>
            <h3>Interface Utilisateur</h3>
            <p>Explorer les outils et la disposition.</p>
          </div>
        </Link>
        
        {/* Orange Theme Card */}
        <Link to="/docs/features/sketcher" className={styles.cardLink}>
          <div className={`${styles.card} ${styles.cardOrange}`}>
            <div className={styles.cardIcon}>
              <Icon path={mdiFeatureSearch} size={1.5} />
            </div>
            <h3>Fonctionnalités</h3>
            <p>Découvrir plus de capacités.</p>
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
    <Layout description="Logiciel moderne d'envoi de G-code et de contrôle pour découpeuses laser basées sur GRBL">
      <main className={styles.pageWrapper}>
        
        <HeroSection />

        <QuickLinks />

        <div className={styles.featuresWrapper}>
          <FeatureSection
            blobClass={styles.blobCyan}
            title="Vos Créations en 3D"
            description="Passez facilement des conceptions 2D aux parcours d'outil 3D. Rayforge génère du G-code pour les découpeuses et graveuses laser 2 axes et 3 axes, comblant le fossé entre les conceptions plates et les objets physiques."
            compatibilityHeading="Compatible avec : Grbl, Smoothieware"
            features={['Visualisation 3D', 'Génération de Parcours d\'Outil', 'Contrôle des Axes']}
            image="/assets/screenshot-3d-closeup.png"
          />

          <FeatureSection
            blobClass={styles.blobOrange}
            title="Ébaucheur 2D Paramétrique"
            description="Créez des conceptions 2D précises basées sur des contraintes directement dans Rayforge. L'ébaucheur paramétrique fournit un ensemble complet d'outils pour créer des formes géométriques et appliquer des contraintes paramétriques."
            features={[
              'Créer des lignes, cercles, arcs et remplissages',
              'Appliquer des contraintes géométriques : coïncident, vertical, horizontal...',
              'Expressions paramétriques pour les dimensions calculées',
            ]}
            image="/assets/screenshot-sketcher.png"
            reverse
          />

          <FeatureSection
            blobClass={styles.blobPurple}
            title="Optimisation Avancée des Parcours"
            description="Gagnez du temps et du matériau avec des algorithmes intelligents de recherche de chemin et une fonctionnalité de mise en page automatique. Rayforge optimise automatiquement l'ordre de découpe pour minimiser le temps de déplacement."
            features={[
              "Ne perdez pas de temps lors de la découpe et de la gravure",
              'Économisez du matériau avec la mise en page et le nesting automatiques',
              'Support natif des arcs avec les commandes G2/G3',
            ]}
            image="/assets/screenshot-optimizer.png"
          />

          <FeatureSection
            blobClass={styles.blobCyan}
            title="Simulateur Intégré Puissant"
            description="Validez votre G-code et simulez le processus de découpe avant de commencer. Notre simulateur intégré vous aide à détecter les problèmes potentiels tôt, vous faisant gagner du temps et des matériaux."
            features={[
              'Aperçu du G-code',
              'Simulation visuelle des parcours de découpe',
              'Aperçu 2D en direct de toutes les opérations pendant l\'édition',
            ]}
            image="/screenshots/main-simulation.png"
            reverse
          />

          <FeatureSection
            blobClass={styles.blobOrange}
            title="Gestion des Matériaux et du Workflow"
            description="Rationalisez votre workflow de découpe laser avec des outils complets de gestion des matériaux, des stocks et des recettes conçus pour améliorer la cohérence et l'efficacité."
            features={[
              'Organiser les matériaux par type et propriétés',
              'Définir les dimensions physiques du stock',
              'Créer des recettes réutilisables pour des résultats cohérents',
            ]}
            image="/assets/screenshot-recipe.png"
          />
        </div>

      </main>
    </Layout>
  );
}
