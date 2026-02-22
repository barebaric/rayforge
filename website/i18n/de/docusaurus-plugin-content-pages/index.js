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
        
        <div className={styles.heroContent}>
          <h1 className={styles.heroTitle}>
            Die weltweit <br />
            <span className={styles.titleGradient}>leistungsstärkste Open-Source</span>
            Laser-Software
          </h1>
          <p className={styles.heroSubtitle}>
            Nahtlose Werkzeuge für kreative Köpfe. Rayforge ist das Open-Source-Power-Tool für Ihren Lasercutter,
            das 2D-CAD, CAM und Maschinensteuerung in einem dynamischen Paket vereint.
          </p>
          <div className={styles.heroCtaButtons}>
            <Link
              to="/docs/getting-started/installation"
              className={styles.buttonPrimary}
            >
              Loslegen
            </Link>
            <a
              href="https://github.com/barebaric/rayforge"
              className={styles.buttonSecondary}
              target="_blank"
              rel="noopener noreferrer"
            >
              Auf GitHub ansehen
            </a>
          </div>
        </div>

        <div className={styles.heroVisuals}>
          <div className={styles.visualBlob}></div>
          
          <div className={styles.screenshotLayer}>
            <img
              src="/screenshots/main-standard.png"
              alt="Rayforge Anwendung Screenshot"
              className={styles.heroImage}
            />
          </div>

          <div className={styles.videoLayer}>
            <div className={styles.videoWrapper}>
              <iframe
                src="https://www.youtube.com/embed/srKXs2p31VY"
                title="Rayforge Einführung"
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
        <Link to="/docs/getting-started/installation" className={styles.cardLink}>
          <div className={`${styles.card} ${styles.cardCyan}`}>
            <div className={styles.cardIcon}>
              <Icon path={mdiRocketLaunch} size={1.5} />
            </div>
            <h3>Erste Schritte</h3>
            <p>Rayforge installieren und konfigurieren.</p>
          </div>
        </Link>
        
        <Link to="/docs/ui/main-window" className={styles.cardLink}>
          <div className={`${styles.card} ${styles.cardPurple}`}>
            <div className={styles.cardIcon}>
              <Icon path={mdiViewDashboard} size={1.5} />
            </div>
            <h3>Benutzeroberfläche</h3>
            <p>Werkzeuge und Layout erkunden.</p>
          </div>
        </Link>
        
        <Link to="/docs/features/sketcher" className={styles.cardLink}>
          <div className={`${styles.card} ${styles.cardOrange}`}>
            <div className={styles.cardIcon}>
              <Icon path={mdiFeatureSearch} size={1.5} />
            </div>
            <h3>Funktionen</h3>
            <p>Weitere Möglichkeiten entdecken.</p>
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
    <Layout description="Moderne G-Code-Sender- und Steuerungssoftware für GRBL-basierte Lasercutter">
      <main className={styles.pageWrapper}>
        
        <HeroSection />

        <QuickLinks />

        <div className={styles.featuresWrapper}>
          <FeatureSection
            blobClass={styles.blobCyan}
            title="Ihre Kreationen in 3D"
            description="Nahtloser Übergang von 2D-Designs zu 3D-Werkzeugpfaden. Rayforge generiert G-Code für 2-Achsen- und 3-Achsen-Lasercutter und -gravierer und überbrückt die Lücke zwischen flachen Designs und physischen Objekten."
            compatibilityHeading="Kompatibel mit: Grbl, Smoothieware"
            features={['3D-Visualisierung', 'Werkzeugpfad-Generierung', 'Achsensteuerung']}
            image="/assets/screenshot-3d-closeup.png"
          />

          <FeatureSection
            blobClass={styles.blobOrange}
            title="Parametrischer 2D-Sketcher"
            description="Erstellen Sie präzise, beschränkungsbasierte 2D-Designs direkt in Rayforge. Der parametrische Sketcher bietet eine vollständige Werkzeugpalette zum Erstellen geometrischer Formen und zum Anwenden parametrischer Beschränkungen."
            features={[
              'Linien, Kreise, Bögen und Füllungen erstellen',
              'Geometrische Beschränkungen anwenden: koinzident, vertikal, horizontal...',
              'Parametrische Ausdrücke für berechnete Abmessungen',
            ]}
            image="/assets/screenshot-sketcher.png"
            reverse
          />

          <FeatureSection
            blobClass={styles.blobPurple}
            title="Erweiterte Pfad-Optimierung"
            description="Sparen Sie Zeit und Material mit intelligenten Pfadsuch-Algorithmen und automatischer Layout-Funktionalität. Rayforge optimiert automatisch die Schneidereihenfolge, um die Bewegungszeit zu minimieren."
            features={[
              'Keine Zeitverschwendung beim Schneiden und Gravieren',
              'Material sparen durch automatisches Layout und Nesting',
              'Native Bogen-Unterstützung mit G2/G3-Befehlen',
            ]}
            image="/assets/screenshot-optimizer.png"
          />

          <FeatureSection
            blobClass={styles.blobCyan}
            title="Leistungsstarker integrierter Simulator"
            description="Validieren Sie Ihren G-Code und simulieren Sie den Schneideprozess, bevor Sie beginnen. Unser integrierter Simulator hilft Ihnen, potenzielle Probleme frühzeitig zu erkennen und spart Zeit und Material."
            features={[
              'G-Code-Vorschau',
              'Visuelle Simulation der Schneidepfade',
              'Live-2D-Vorschau aller Operationen während der Bearbeitung',
            ]}
            image="/screenshots/main-simulation.png"
            reverse
          />

          <FeatureSection
            blobClass={styles.blobOrange}
            title="Material- und Workflow-Verwaltung"
            description="Optimieren Sie Ihren Lasercutting-Workflow mit umfassenden Werkzeugen zur Verwaltung von Materialien, Rohlingen und Rezepten, die für verbesserte Konsistenz und Effizienz entwickelt wurden."
            features={[
              'Materialien nach Typ und Eigenschaften organisieren',
              'Physische Rohlingsabmessungen definieren',
              'Wiederverwendbare Rezepte für konsistente Ergebnisse erstellen',
            ]}
            image="/assets/screenshot-recipe.png"
          />
        </div>

      </main>
    </Layout>
  );
}
