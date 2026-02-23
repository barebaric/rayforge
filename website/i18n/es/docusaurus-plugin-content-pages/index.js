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
            Herramientas fluidas para mentes imaginativas. Rayforge es la herramienta de código abierto para tu cortador láser,
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
    <Layout description="Software moderno de envío de G-code y control para cortadores láser basados en GRBL">
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
            title="Dibujador 2D Paramétrico"
            description="Crea diseños 2D precisos basados en restricciones directamente en Rayforge. El dibujador paramétrico proporciona un conjunto completo de herramientas para crear formas geométricas y aplicar restricciones paramétricas."
            features={[
              'Crear líneas, círculos, arcos y rellenos',
              'Aplicar restricciones geométricas: coincidente, vertical, horizontal...',
              'Expresiones paramétricas para dimensiones calculadas',
            ]}
            image="/assets/screenshot-sketcher.png"
            reverse
          />

          <FeatureSection
            blobClass={styles.blobPurple}
            title="Optimización Avanzada de Trayectorias"
            description="Ahorra tiempo y material con algoritmos inteligentes de búsqueda de rutas y funcionalidad de diseño automático. Rayforge optimiza automáticamente el orden de corte para minimizar el tiempo de desplazamiento."
            features={[
              "No pierdas tiempo al cortar y grabar",
              'Ahorra material con diseño y anidamiento automáticos',
              'Soporte nativo de arcos con comandos G2/G3',
            ]}
            image="/assets/screenshot-optimizer.png"
          />

          <FeatureSection
            blobClass={styles.blobCyan}
            title="Simulador Integrado Potente"
            description="Valida tu G-code y simula el proceso de corte antes de comenzar. Nuestro simulador integrado te ayuda a detectar problemas potenciales temprano, ahorrándote tiempo y materiales."
            features={[
              'Vista previa de G-code',
              'Simulación visual de trayectorias de corte',
              'Vista previa 2D en vivo de todas las operaciones mientras editas',
            ]}
            image="/screenshots/main-simulation.png"
            reverse
          />

          <FeatureSection
            blobClass={styles.blobOrange}
            title="Gestión de Materiales y Flujos de Trabajo"
            description="Optimiza tu flujo de trabajo de corte láser con herramientas integrales de gestión de materiales, stock y recetas diseñadas para mejorar la consistencia y eficiencia."
            features={[
              'Organizar materiales por tipo y propiedades',
              'Definir dimensiones físicas del stock',
              'Crear recetas reutilizables para resultados consistentes',
            ]}
            image="/assets/screenshot-recipe.png"
          />
        </div>

      </main>
    </Layout>
  );
}
