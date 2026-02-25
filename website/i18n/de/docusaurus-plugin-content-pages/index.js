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
        
        <div className={styles.heroContent}>
          <h1 className={styles.heroTitle}>
            <span className={styles.titleGradient}>Die weltweit leistungsstärkste Open-Source</span> Laser-Software
          </h1>
          <p className={styles.heroSubtitle}>
            Nahtlose Werkzeuge für kreative Köpfe. Rayforge ist das komplette Kreativstudio für deinen Lasercutter,
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
              src={useBaseUrl('/screenshots/main-standard.png')}
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
    <Layout description="Gestalte und erschaffe mit deinem Lasercutter - das komplette Kreativstudio für Maker, Künstler und Handwerker">
      <main className={styles.pageWrapper}>
        
        <HeroSection />

        <QuickLinks />

        <div className={styles.featuresWrapper}>
          <FeatureSection
            blobClass={styles.blobCyan}
            title="Deine Kreationen in 3D"
            description="Nahtloser Übergang von 2D-Designs zu 3D-Werkzeugpfaden. Rayforge generiert G-Code für 2-Achsen- und 3-Achsen-Lasercutter und -gravierer und überbrückt die Lücke zwischen flachen Designs und physischen Objekten."
            compatibilityHeading="Kompatibel mit: Grbl, Smoothieware"
            features={['3D-Visualisierung', 'Werkzeugpfad-Generierung', 'Achsensteuerung']}
            image="/assets/screenshot-3d-closeup.png"
          />

          <FeatureSection
            blobClass={styles.blobOrange}
            title="Gestalte deine eigenen Formen"
            description="Erstelle benutzerdefinierte Designs direkt in Rayforge. Die integrierten Zeichenwerkzeuge ermöglichen es dir, deine Ideen zu skizzieren, zu formen und zu verfeinern, ohne separate Designsoftware zu benötigen."
            features={[
              'Zeichne Linien, Kreise, Kurven und gefüllte Formen',
              'Richte alles perfekt aus',
              'Definiere Abmessungen, die sich automatisch aktualisieren',
            ]}
            image="/assets/screenshot-sketcher.png"
            reverse
          />

          <FeatureSection
            blobClass={styles.blobPurple}
            title="Intelligente Schneidepfade"
            description="Arbeite schneller und verschwende weniger Material. Rayforge ermittelt die effizienteste Methode, dein Design zu schneiden, damit dein Laser weniger Zeit mit Bewegen und mehr Zeit mit Erschaffen verbringt."
            features={[
              "Schnelleres Schneiden durch optimierte Bewegungen",
              'Material sparen durch automatisches Layout',
              'Glatte, präzise Kurven',
            ]}
            image="/assets/screenshot-optimizer.png"
          />

          <FeatureSection
            blobClass={styles.blobCyan}
            title="Vorschau Vor Dem Schneiden"
            description="Sieh dein Design virtuell zum Leben erwachen, bevor du dich auf das echte Material festlegst. Erkenne potenzielle Probleme früh und spare Zeit und Material."
            features={[
              'Sieh genau, wie sich dein Laser bewegen wird',
              'Fehler erkennen bevor du schneidest',
              'Live-Vorschau, die sich während des Designs aktualisiert',
            ]}
            image="/screenshots/main-simulation.png"
            reverse
          />

          <FeatureSection
            blobClass={styles.blobOrange}
            title="Material- und Workflow-Verwaltung"
            description="Speichere deine Lieblingseinstellungen und erhalte konsistente Ergebnisse in all deinen Projekten. Ob du Holz, Acryl oder Leder schneidest - Rayforge merkt sich, was am besten funktioniert."
            features={[
              'Einstellungen für verschiedene Materialien speichern',
              'Wiederverwendbare Rezepte erstellen',
            ]}
            image="/assets/screenshot-recipe.png"
          />
        </div>

      </main>
    </Layout>
  );
}
