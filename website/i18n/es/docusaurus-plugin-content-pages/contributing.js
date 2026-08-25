import React from 'react';
import Layout from '@theme/Layout';
import Link from '@docusaurus/Link';
import Icon from '@mdi/react';
import {
  mdiBugOutline,
  mdiLightbulbOnOutline,
  mdiSourcePull,
  mdiBookOpenPageVariantOutline,
  mdiHandCoinOutline,
  mdiGithub,
  mdiYoutube,
  mdiPlayCircleOutline,
  mdiStarOutline,
  mdiFire,
} from '@mdi/js';
import styles from '@site/src/pages/contributing.module.css';

const wishlistTopics = [
  'Configuración del eje rotatorio',
  'Generador de piezas con IA',
  'Calibración de cámara',
  'Flujo de trabajo Print & Cut',
  'Pruebas de materiales',
];

const quickActions = [
  {
    title: 'Reportar un Error',
    description: 'Abre un issue con los pasos para reproducirlo y lo que esperabas.',
    href: 'https://github.com/barebaric/rayforge/issues/new',
    icon: mdiBugOutline,
    iconClass: styles.iconCyan,
  },
  {
    title: 'Sugerir una Función',
    description: 'Comparte un caso de uso y cómo se ve el éxito para ti.',
    href: 'https://github.com/barebaric/rayforge/issues/new?labels=enhancement',
    icon: mdiLightbulbOnOutline,
    iconClass: styles.iconOrange,
  },
  {
    title: 'Enviar Código',
    description: 'Sigue la guía para desarrolladores y envía un pull request.',
    to: '/docs/developer/getting-started',
    icon: mdiSourcePull,
    iconClass: styles.iconPurple,
  },
  {
    title: 'Mejorar la Documentación',
    description: 'Corrige erratas, añade ejemplos y haz que la documentación sea más fácil de seguir.',
    to: '/docs/getting-started/installation',
    icon: mdiBookOpenPageVariantOutline,
    iconClass: styles.iconCyan,
  },
];

export default function Contributing() {
  return (
    <Layout
      title="Contribuir"
      description="Aprende cómo contribuir a Rayforge: reporta errores, sugiere funciones, envía código, crea tutoriales de vídeo, mejora la documentación o apoya el proyecto económicamente."
    >
      <main className={styles.pageWrapper}>
        <section className={styles.hero}>
          <div className={styles.heroInner}>
            <div className={styles.heroContent}>
              <h1 className={styles.heroTitle}>
                Contribuir a{' '}
                <span className={styles.heroTitleGradient}>Rayforge</span>
              </h1>
              <p className={styles.heroSubtitle}>
                Ayuda a mejorar Rayforge: reporta errores, sugiere funciones,
                envía código, mejora la documentación, crea tutoriales o apoya
                el proyecto económicamente.
              </p>
              <div className={styles.heroCtas}>
                <a
                  href="https://www.patreon.com/c/knipknap"
                  className={`rfButton rfButtonOrange ${styles.heroCtaButton}`}
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  <Icon path={mdiHandCoinOutline} size={0.9} />
                  <span>Apoyar en Patreon</span>
                </a>
                <a
                  href="https://github.com/barebaric/rayforge/issues/new"
                  className={`rfButton rfButtonDownload ${styles.heroCtaButton}`}
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  <Icon path={mdiBugOutline} size={0.9} />
                  <span>Reportar un Error</span>
                </a>
                <Link
                  to="/docs/developer/getting-started"
                  className={`rfButton rfButtonPurple ${styles.heroCtaButton}`}
                >
                  <Icon path={mdiSourcePull} size={0.9} />
                  <span>Empezar a Contribuir</span>
                </Link>
              </div>
            </div>

            <div className={styles.heroPanel}>
              <div className={styles.panelHeader}>
                <div className={styles.panelBadge}>
                  <Icon path={mdiGithub} size={0.85} />
                  <span>GitHub</span>
                </div>
                <h2 className={styles.panelTitle}>Comunidad y Soporte</h2>
              </div>
              <div className={styles.panelLinks}>
                <a
                  href="https://github.com/barebaric/rayforge/issues"
                  className={styles.panelLink}
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  <span className={styles.panelLinkLabel}>Reportar problemas</span>
                  <span className={styles.panelLinkMeta}>GitHub Issues</span>
                </a>
                <a
                  href="https://github.com/barebaric/rayforge"
                  className={styles.panelLink}
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  <span className={styles.panelLinkLabel}>Explorar el código</span>
                  <span className={styles.panelLinkMeta}>Repositorio de GitHub</span>
                </a>
                <Link to="/sponsor" className={styles.panelLink}>
                  <span className={styles.panelLinkLabel}>
                    Convertirse en Patrocinador
                  </span>
                  <span className={styles.panelLinkMeta}>Ayúdanos a Mejorar</span>
                </Link>
              </div>
            </div>
          </div>
        </section>

        <section className={styles.section}>
          <div className={styles.sectionInner}>
            <h2 className={styles.sectionTitle}>Causa el Mayor Impacto</h2>
            <p className={styles.lead}>
              Algunas contribuciones importan más que otras. Ahora mismo, nada
              ayuda a Rayforge a crecer tanto como los tutoriales de vídeo — y
              tu generosidad mantiene vivo el proyecto.
            </p>

            <div className={styles.impactGrid}>
              <div
                className={`${styles.impactCard} ${styles.impactTutorial}`}
                id="video-tutorials"
              >
                <div
                  className={`${styles.impactBadge} ${styles.impactBadgeTutorial}`}
                >
                  <Icon path={mdiStarOutline} size={0.8} />
                  <span>Muy Solicitado</span>
                </div>
                <div className={styles.impactCardHeader}>
                  <div className={`${styles.blockIcon} ${styles.iconRed}`}>
                    <Icon path={mdiYoutube} size={1.1} />
                  </div>
                  <h3 className={styles.impactCardTitle}>
                    Crear Tutoriales de Vídeo
                  </h3>
                </div>
                <p className={styles.impactCardBody}>
                  Los vídeos son como la mayoría de la gente descubre Rayforge
                  — y los tutoriales terminados se muestran en la portada con
                  tu nombre y un enlace a tu canal.
                </p>
                <ol className={styles.steps}>
                  <li className={styles.step}>
                    Elige un tema de abajo — o propón uno propio.
                  </li>
                  <li className={styles.step}>
                    Graba una captura de pantalla corta con narración, súbela
                    a YouTube y comparte el enlace en{' '}
                    <a
                      href="https://discord.gg/sTHNdTtpQJ"
                      target="_blank"
                      rel="noopener noreferrer"
                    >
                      Discord
                    </a>{' '}
                    o en{' '}
                    <a
                      href="https://github.com/barebaric/rayforge/discussions"
                      target="_blank"
                      rel="noopener noreferrer"
                    >
                      GitHub Discussions
                    </a>{' '}
                    para reclamar tu lugar.
                  </li>
                </ol>
                <div className={styles.wishlist}>
                  <div className={styles.wishlistTitle}>
                    <Icon path={mdiFire} size={0.85} />
                    <span>Lista de deseos — reclama un tema</span>
                  </div>
                  <div className={styles.wishlistChips}>
                    {wishlistTopics.map((topic) => (
                      <span className={styles.wishlistChip} key={topic}>
                        {topic}
                      </span>
                    ))}
                  </div>
                </div>
                <a
                  href="https://discord.gg/sTHNdTtpQJ"
                  className={`rfButton rfButtonOrange ${styles.impactCta}`}
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  <Icon path={mdiPlayCircleOutline} size={0.9} />
                  <span>Comparte tu Tutorial</span>
                </a>
              </div>

              <div
                className={`${styles.impactCard} ${styles.impactSupport}`}
              >
                <div
                  className={`${styles.impactBadge} ${styles.impactBadgeSupport}`}
                >
                  <Icon path={mdiHandCoinOutline} size={0.8} />
                  <span>Mantiene el Proyecto Vivo</span>
                </div>
                <div className={styles.impactCardHeader}>
                  <div className={`${styles.blockIcon} ${styles.iconPurple}`}>
                    <Icon path={mdiHandCoinOutline} size={1.1} />
                  </div>
                  <h3 className={styles.impactCardTitle}>
                    Apoyar Económicamente
                  </h3>
                </div>
                <p className={styles.impactCardBody}>
                  Rayforge es gratuito y seguirá siéndolo. El dinero de Patreon
                  y de los patrocinios paga servidores, hardware de pruebas y
                  tiempo de desarrollo — mantiene el proyecto en movimiento.
                </p>
                <div className={styles.impactLinks}>
                  <a
                    href="https://www.patreon.com/c/knipknap"
                    className={`rfButton rfButtonOrange ${styles.impactCta}`}
                    target="_blank"
                    rel="noopener noreferrer"
                  >
                    <Icon path={mdiHandCoinOutline} size={0.9} />
                    <span>Apoyar en Patreon</span>
                  </a>
                  <Link
                    to="/sponsor"
                    className={`rfButton rfButtonPurple ${styles.impactCta}`}
                  >
                    <Icon path={mdiStarOutline} size={0.9} />
                    <span>Convertirse en Patrocinador</span>
                  </Link>
                </div>
              </div>
            </div>
          </div>
        </section>

        <section className={styles.section}>
          <div className={styles.sectionInner}>
            <h2 className={styles.sectionTitle}>Acciones Rápidas</h2>
            <div className={styles.cardGrid}>
              {quickActions.map((action) => {
                const cardInner = (
                  <>
                    <div className={`${styles.cardIcon} ${action.iconClass}`}>
                      <Icon path={action.icon} size={1.1} />
                    </div>
                    <div className={styles.cardBody}>
                      <h3 className={styles.cardTitle}>{action.title}</h3>
                      <p className={styles.cardDescription}>
                        {action.description}
                      </p>
                    </div>
                  </>
                );

                if (action.to) {
                  return (
                    <Link key={action.title} to={action.to} className={styles.card}>
                      {cardInner}
                    </Link>
                  );
                }

                return (
                  <a
                    key={action.title}
                    href={action.href}
                    className={styles.card}
                    target="_blank"
                    rel="noopener noreferrer"
                  >
                    {cardInner}
                  </a>
                );
              })}
            </div>
          </div>
        </section>

        <section className={styles.section}>
          <div className={styles.sectionInner}>
            <h2 className={styles.sectionTitle}>Sobre Esta Documentación</h2>
            <p className={styles.lead}>
              Esta documentación está diseñada para los usuarios finales de
              Rayforge. Para documentación de desarrolladores, empieza aquí:{' '}
              <Link to="/docs/developer/getting-started">Documentación para Desarrolladores</Link>.
            </p>
          </div>
        </section>
      </main>
    </Layout>
  );
}
