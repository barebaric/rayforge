import React from 'react';
import Layout from '@theme/Layout';
import Link from '@docusaurus/Link';
import useBaseUrl from '@docusaurus/useBaseUrl';
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
            Des outils fluides pour les esprits imaginatifs. Rayforge est le studio créatif complet pour votre découpeuse laser,
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
              src={useBaseUrl('/screenshots/main-standard.png')}
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
          <img src={useBaseUrl(image)} alt={title} />
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
    <Layout description="Concevez et créez avec votre découpeuse laser - le studio créatif complet pour les créateurs, artistes et artisans">
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
            title="Concevez Vos Propres Formes"
            description="Créez des designs personnalisés directement dans Rayforge. Les outils de dessin intégrés vous permettent d'esquisser, de façonner et d'affiner vos idées sans avoir besoin d'un logiciel de conception séparé."
            features={[
              'Dessinez des lignes, cercles, courbes et formes remplies',
              'Alignez tout parfaitement',
              'Définissez des dimensions qui se mettent à jour automatiquement',
            ]}
            image="/assets/screenshot-sketcher.png"
            reverse
          />

          <FeatureSection
            blobClass={styles.blobPurple}
            title="Parcours de Découpe Intelligents"
            description="Travaillez plus vite et gaspillez moins de matériau. Rayforge détermine la manière la plus efficace de découper votre design, pour que votre laser passe moins de temps à se déplacer et plus de temps à créer."
            features={[
              "Découpe plus rapide avec des mouvements optimisés",
              'Économisez du matériau avec la mise en page automatique',
              'Courbes fluides et précises',
            ]}
            image="/assets/screenshot-optimizer.png"
          />

          <FeatureSection
            blobClass={styles.blobCyan}
            title="Prévisualisez Avant de Découper"
            description="Regardez votre design prendre vie virtuellement avant de vous engager sur le vrai matériau. Détectez les problèmes potentiels tôt et évitez de perdre du temps et des matériaux."
            features={[
              'Voyez exactement comment votre laser va se déplacer',
              'Repérez les erreurs avant de découper',
              'Aperçu en direct qui se met à jour pendant la conception',
            ]}
            image="/screenshots/main-simulation.png"
            reverse
          />

          <FeatureSection
            blobClass={styles.blobOrange}
            title="Gestion des Matériaux et du Workflow"
            description="Sauvegardez vos réglages favoris et obtenez des résultats cohérents pour tous vos projets. Que vous découpiez du bois, de l'acrylique ou du cuir, Rayforge se souvient de ce qui fonctionne le mieux."
            features={[
              'Enregistrez les réglages pour différents matériaux',
              'Créez des recettes réutilisables',
            ]}
            image="/assets/screenshot-recipe.png"
          />
        </div>

      </main>
    </Layout>
  );
}
