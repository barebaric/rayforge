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
            El <span className={styles.titleGradient}>Software Láser de Código Abierto Más Potente</span> del Mundo
          </h1>
          <p className={styles.heroSubtitle}>
            Herramientas fluidas para mentes imaginativas. Rayforge es el estudio creativo completo para tu cortador láser,
            combinando CAD 2D, CAM y control de máquina en un paquete vibrante.
          </p>
          <div className={styles.heroCtaButtons}>
            <Link
              to="/docs/getting-started/installation"
              className={styles.buttonPrimary}
            >
              Comenzar
            </Link>
            <a
              href="https://github.com/barebaric/rayforge"
              className={styles.buttonSecondary}
              target="_blank"
              rel="noopener noreferrer"
            >
              Ver en GitHub
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
              alt="Captura de pantalla de Rayforge"
              className={styles.heroImage}
            />
          </div>

          {/* Floating Front Layer: Video */}
          <div className={styles.videoLayer}>
            <div className={styles.videoWrapper}>
              <iframe
                src="https://www.youtube.com/embed/srKXs2p31VY"
                title="Introducción a Rayforge"
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
            <h3>Primeros Pasos</h3>
            <p>Instala y configura Rayforge.</p>
          </div>
        </Link>
        
        {/* Purple Theme Card */}
        <Link to="/docs/ui/main-window" className={styles.cardLink}>
          <div className={`${styles.card} ${styles.cardPurple}`}>
            <div className={styles.cardIcon}>
              <Icon path={mdiViewDashboard} size={1.5} />
            </div>
            <h3>Interfaz de Usuario</h3>
            <p>Explora las herramientas y el diseño.</p>
          </div>
        </Link>
        
        {/* Orange Theme Card */}
        <Link to="/docs/features/sketcher" className={styles.cardLink}>
          <div className={`${styles.card} ${styles.cardOrange}`}>
            <div className={styles.cardIcon}>
              <Icon path={mdiFeatureSearch} size={1.5} />
            </div>
            <h3>Características</h3>
            <p>Descubre más capacidades.</p>
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
    <Layout description="Diseña y crea con tu cortador láser - el estudio creativo completo para makers, artistas y artesanos">
      <main className={styles.pageWrapper}>
        
        <HeroSection />

        <QuickLinks />

        <div className={styles.featuresWrapper}>
          <FeatureSection
            blobClass={styles.blobCyan}
            title="Tus Creaciones en 3D"
            description="Transición fluida de diseños 2D a trayectorias 3D. Rayforge genera G-code para cortadores y grabadores láser de 2 y 3 ejes, cerrando la brecha entre diseños planos y objetos físicos."
            compatibilityHeading="Compatible con: Grbl, Smoothieware"
            features={['Visualización 3D', 'Generación de Trayectorias', 'Control de Ejes']}
            image="/assets/screenshot-3d-closeup.png"
          />

          <FeatureSection
            blobClass={styles.blobOrange}
            title="Diseña Tus Propias Formas"
            description="Crea diseños personalizados directamente en Rayforge. Las herramientas de dibujo integradas te permiten bocetar, moldear y refinar tus ideas sin necesidad de software de diseño separado."
            features={[
              'Dibuja líneas, círculos, curvas y formas rellenas',
              'Alinea todo perfectamente',
              'Define dimensiones que se actualizan automáticamente',
            ]}
            image="/assets/screenshot-sketcher.png"
            reverse
          />

          <FeatureSection
            blobClass={styles.blobPurple}
            title="Trayectorias de Corte Inteligentes"
            description="Trabaja más rápido y desperdicia menos material. Rayforge descubre la forma más eficiente de cortar tu diseño, para que tu láser pase menos tiempo moviéndose y más tiempo creando."
            features={[
              "Corte más rápido con movimientos optimizados",
              'Ahorra material con diseño automático',
              'Curvas suaves y precisas',
            ]}
            image="/assets/screenshot-optimizer.png"
          />

          <FeatureSection
            blobClass={styles.blobCyan}
            title="Previsualiza Antes de Cortar"
            description="Mira tu diseño cobrar vida virtualmente antes de comprometerte con el material real. Detecta problemas potenciales temprano y ahorra tiempo y materiales."
            features={[
              'Ve exactamente cómo se moverá tu láser',
              'Detecta errores antes de cortar',
              'Vista previa en vivo que se actualiza mientras diseñas',
            ]}
            image="/screenshots/main-simulation.png"
            reverse
          />

          <FeatureSection
            blobClass={styles.blobOrange}
            title="Gestión de Materiales y Flujos de Trabajo"
            description="Guarda tus configuraciones favoritas y obtén resultados consistentes en todos tus proyectos. Ya sea que cortes madera, acrílico o cuero, Rayforge recuerda lo que funciona mejor."
            features={[
              'Guarda configuraciones para diferentes materiales',
              'Crea recetas reutilizables',
            ]}
            image="/assets/screenshot-recipe.png"
          />
        </div>

      </main>
    </Layout>
  );
}
