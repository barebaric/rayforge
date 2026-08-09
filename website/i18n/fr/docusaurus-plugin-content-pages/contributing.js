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
  mdiVideoOutline,
  mdiPlayCircleOutline,
  mdiStarOutline,
  mdiFire,
} from '@mdi/js';
import styles from '@site/src/pages/contributing.module.css';

const wishlistTopics = [
  'Votre première gravure',
  "Configuration de l'axe rotatif",
  'Générateur de pièces IA',
  'Calibrage de la caméra',
  'Flux de travail Print & Cut',
  'Tests de matériaux',
];

const quickActions = [
  {
    title: 'Signaler un Bug',
    description: 'Ouvrez un issue avec les étapes pour reproduire et ce que vous attendiez.',
    href: 'https://github.com/barebaric/rayforge/issues/new',
    icon: mdiBugOutline,
    iconClass: styles.iconCyan,
  },
  {
    title: 'Suggérer une Fonctionnalité',
    description: 'Partagez un cas d\'utilisation et ce à quoi ressemble le succès pour vous.',
    href: 'https://github.com/barebaric/rayforge/issues/new?labels=enhancement',
    icon: mdiLightbulbOnOutline,
    iconClass: styles.iconOrange,
  },
  {
    title: 'Soumettre du Code',
    description: 'Suivez le guide développeur et envoyez une pull request.',
    to: '/docs/developer/getting-started',
    icon: mdiSourcePull,
    iconClass: styles.iconPurple,
  },
  {
    title: 'Améliorer la Documentation',
    description: 'Corrigez les fautes, ajoutez des exemples et rendez la doc plus facile à suivre.',
    to: '/docs/getting-started/installation',
    icon: mdiBookOpenPageVariantOutline,
    iconClass: styles.iconCyan,
  },
  {
    title: 'Créer des Tutoriels Vidéo',
    description:
      'Apprenez Rayforge au monde — les tutoriels terminés sont mis en avant sur la page d\'accueil.',
    href: '#video-tutorials',
    icon: mdiVideoOutline,
    iconClass: styles.iconRed,
    featured: true,
  },
];

export default function Contributing() {
  return (
    <Layout
      title="Contribuer"
      description="Apprendre comment contribuer à Rayforge : signaler des bugs, suggérer des fonctionnalités, soumettre du code, créer des tutoriels vidéo, améliorer la doc ou soutenir le projet financièrement."
    >
      <main className={styles.pageWrapper}>
        <section className={styles.hero}>
          <div className={styles.heroInner}>
            <div className={styles.heroContent}>
              <h1 className={styles.heroTitle}>
                Contribuer à{' '}
                <span className={styles.heroTitleGradient}>Rayforge</span>
              </h1>
              <p className={styles.heroSubtitle}>
                Aidez à améliorer Rayforge : signalez des bugs, suggérez des
                fonctionnalités, soumettez du code, améliorez la doc, créez
                des tutoriels ou soutenez le projet financièrement.
              </p>
              <div className={styles.heroCtas}>
                <a
                  href="https://www.patreon.com/c/knipknap"
                  className={`rfButton rfButtonOrange ${styles.heroCtaButton}`}
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  <Icon path={mdiHandCoinOutline} size={0.9} />
                  <span>Soutenir sur Patreon</span>
                </a>
                <a
                  href="https://github.com/barebaric/rayforge/issues/new"
                  className={`rfButton rfButtonDownload ${styles.heroCtaButton}`}
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  <Icon path={mdiBugOutline} size={0.9} />
                  <span>Signaler un Bug</span>
                </a>
                <Link
                  to="/docs/developer/getting-started"
                  className={`rfButton rfButtonPurple ${styles.heroCtaButton}`}
                >
                  <Icon path={mdiSourcePull} size={0.9} />
                  <span>Commencer à Contribuer</span>
                </Link>
              </div>
            </div>

            <div className={styles.heroPanel}>
              <div className={styles.panelHeader}>
                <div className={styles.panelBadge}>
                  <Icon path={mdiGithub} size={0.85} />
                  <span>GitHub</span>
                </div>
                <h2 className={styles.panelTitle}>Communauté et Support</h2>
              </div>
              <div className={styles.panelLinks}>
                <a
                  href="https://github.com/barebaric/rayforge/issues"
                  className={styles.panelLink}
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  <span className={styles.panelLinkLabel}>Signaler des problèmes</span>
                  <span className={styles.panelLinkMeta}>Issues GitHub</span>
                </a>
                <a
                  href="https://github.com/barebaric/rayforge"
                  className={styles.panelLink}
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  <span className={styles.panelLinkLabel}>Parcourir le code source</span>
                  <span className={styles.panelLinkMeta}>Dépôt GitHub</span>
                </a>
                <Link to="/sponsor" className={styles.panelLink}>
                  <span className={styles.panelLinkLabel}>
                    Devenir Sponsor
                  </span>
                  <span className={styles.panelLinkMeta}>Aidez-nous à nous améliorer</span>
                </Link>
              </div>
            </div>
          </div>
        </section>

        <section className={styles.section}>
          <div className={styles.sectionInner}>
            <h2 className={styles.sectionTitle}>Ayez le Plus d'Impact</h2>
            <p className={styles.lead}>
              Certaines contributions comptent plus que d'autres. Actuellement,
              rien n'aide Rayforge à grandir autant que les tutoriels vidéo —
              et votre générosité maintient le projet en vie.
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
                  <span>Très Demandé</span>
                </div>
                <div className={styles.impactCardHeader}>
                  <div className={`${styles.blockIcon} ${styles.iconRed}`}>
                    <Icon path={mdiYoutube} size={1.1} />
                  </div>
                  <h3 className={styles.impactCardTitle}>
                    Créer des Tutoriels Vidéo
                  </h3>
                </div>
                <p className={styles.impactCardBody}>
                  C'est par les vidéos que la plupart des gens découvrent
                  Rayforge — et les tutoriels terminés sont mis en avant sur la
                  page d'accueil avec votre nom et un lien vers votre chaîne.
                </p>
                <ol className={styles.steps}>
                  <li className={styles.step}>
                    Choisissez un sujet ci-dessous — ou proposez le vôtre.
                  </li>
                  <li className={styles.step}>
                    Enregistrez une courte capture d'écran commentée,
                    téléversez-la sur YouTube et partagez le lien sur{' '}
                    <a
                      href="https://discord.gg/sTHNdTtpQJ"
                      target="_blank"
                      rel="noopener noreferrer"
                    >
                      Discord
                    </a>{' '}
                    ou dans les{' '}
                    <a
                      href="https://github.com/barebaric/rayforge/discussions"
                      target="_blank"
                      rel="noopener noreferrer"
                    >
                      Discussions GitHub
                    </a>{' '}
                    pour réserver votre place.
                  </li>
                </ol>
                <div className={styles.wishlist}>
                  <div className={styles.wishlistTitle}>
                    <Icon path={mdiFire} size={0.85} />
                    <span>Liste de souhaits — réclamez un sujet</span>
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
                  <span>Partagez votre Tutoriel</span>
                </a>
              </div>

              <div
                className={`${styles.impactCard} ${styles.impactSupport}`}
              >
                <div
                  className={`${styles.impactBadge} ${styles.impactBadgeSupport}`}
                >
                  <Icon path={mdiHandCoinOutline} size={0.8} />
                  <span>Maintient le Projet en Vie</span>
                </div>
                <div className={styles.impactCardHeader}>
                  <div className={`${styles.blockIcon} ${styles.iconPurple}`}>
                    <Icon path={mdiHandCoinOutline} size={1.1} />
                  </div>
                  <h3 className={styles.impactCardTitle}>
                    Soutenir Financièrement
                  </h3>
                </div>
                <p className={styles.impactCardBody}>
                  Rayforge est gratuit et le restera. L'argent de Patreon et
                  des parrainages finance les serveurs, le matériel de test et
                  le temps de développement — il fait avancer le projet.
                </p>
                <div className={styles.impactLinks}>
                  <a
                    href="https://www.patreon.com/c/knipknap"
                    className={`rfButton rfButtonOrange ${styles.impactCta}`}
                    target="_blank"
                    rel="noopener noreferrer"
                  >
                    <Icon path={mdiHandCoinOutline} size={0.9} />
                    <span>Soutenir sur Patreon</span>
                  </a>
                  <Link
                    to="/sponsor"
                    className={`rfButton rfButtonPurple ${styles.impactCta}`}
                  >
                    <Icon path={mdiStarOutline} size={0.9} />
                    <span>Devenir Sponsor</span>
                  </Link>
                </div>
              </div>
            </div>
          </div>
        </section>

        <section className={styles.section}>
          <div className={styles.sectionInner}>
            <h2 className={styles.sectionTitle}>Actions Rapides</h2>
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

                if (action.featured) {
                  return (
                    <a
                      key={action.title}
                      href={action.href}
                      className={`${styles.card} ${styles.featuredCard}`}
                    >
                      {cardInner}
                      <span className={styles.featuredCardCta}>
                        <Icon path={mdiPlayCircleOutline} size={0.85} />
                        <span>Être Mis en Avant</span>
                      </span>
                    </a>
                  );
                }

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
            <h2 className={styles.sectionTitle}>Manières de Contribuer</h2>
            <p className={styles.lead}>
              Nous accueillons toutes sortes de contributions. Chaque rapport
              de bug, chaque PR et chaque correction de documentation rend
              Rayforge meilleur pour tous.
            </p>

            <div className={styles.twoCol}>
              <div className={styles.block}>
                <div className={styles.blockHeader}>
                  <div className={`${styles.blockIcon} ${styles.iconCyan}`}>
                    <Icon path={mdiBugOutline} size={0.95} />
                  </div>
                  <h3 className={styles.blockTitle}>Signaler des Bugs</h3>
                </div>
                <ol className={styles.steps}>
                  <li className={styles.step}>
                    Consultez les{' '}
                    <a
                      href="https://github.com/barebaric/rayforge/issues"
                      target="_blank"
                      rel="noopener noreferrer"
                    >
                      problèmes existants
                    </a>{' '}
                    pour éviter les doublons.
                  </li>
                  <li className={styles.step}>
                    Créez un{' '}
                    <a
                      href="https://github.com/barebaric/rayforge/issues/new"
                      target="_blank"
                      rel="noopener noreferrer"
                    >
                      nouveau problème
                    </a>{' '}
                    avec une description claire, les étapes pour reproduire,
                    le comportement attendu vs. actuel, les informations
                    système et des captures d'écran si applicable.
                  </li>
                </ol>
              </div>

              <div className={styles.block}>
                <div className={styles.blockHeader}>
                  <div className={`${styles.blockIcon} ${styles.iconOrange}`}>
                    <Icon path={mdiLightbulbOnOutline} size={0.95} />
                  </div>
                  <h3 className={styles.blockTitle}>Suggérer des Fonctionnalités</h3>
                </div>
                <ol className={styles.steps}>
                  <li className={styles.step}>
                    Consultez les{' '}
                    <a
                      href="https://github.com/barebaric/rayforge/issues?q=is%3Aissue+label%3Aenhancement"
                      target="_blank"
                      rel="noopener noreferrer"
                    >
                      demandes de fonctionnalités existantes
                    </a>
                    .
                  </li>
                  <li className={styles.step}>
                    Ouvrez une demande de fonctionnalité décrivant l'idée, le
                    cas d'utilisation, les avantages et (optionnellement) une
                    approche possible.
                  </li>
                </ol>
              </div>

              <div className={styles.block}>
                <div className={styles.blockHeader}>
                  <div className={`${styles.blockIcon} ${styles.iconPurple}`}>
                    <Icon path={mdiSourcePull} size={0.95} />
                  </div>
                  <h3 className={styles.blockTitle}>Soumettre du Code</h3>
                </div>
                <p className={styles.blockBody}>
                  Pour des informations détaillées sur la soumission de
                  contributions de code, veuillez consulter le guide{' '}
                  <Link to="/docs/developer/getting-started">
                    Documentation Développeur – Pour Commencer
                  </Link>{' '}
                  .
                </p>
              </div>

              <div className={styles.block}>
                <div className={styles.blockHeader}>
                  <div className={`${styles.blockIcon} ${styles.iconCyan}`}>
                    <Icon path={mdiBookOpenPageVariantOutline} size={0.95} />
                  </div>
                  <h3 className={styles.blockTitle}>Améliorer la Documentation</h3>
                </div>
                <ul className={styles.bullets}>
                  <li>Corriger les fautes de frappe ou les explications peu claires</li>
                  <li>Ajouter des exemples et des captures d'écran</li>
                  <li>Améliorer l'organisation</li>
                  <li>Traduire dans d'autres langues</li>
                </ul>
                <p className={styles.blockBody}>
                  Vous pouvez cliquer sur le bouton « modifier cette page » sur
                  n'importe quelle page de documentation, puis soumettre des PR
                  de la même manière que les contributions de code.
                </p>
              </div>
            </div>
          </div>
        </section>

        <section className={styles.section}>
          <div className={styles.sectionInner}>
            <h2 className={styles.sectionTitle}>À Propos de Cette Documentation</h2>
            <p className={styles.lead}>
              Cette documentation est conçue pour les utilisateurs finaux de
              Rayforge. Pour la documentation développeur, commencez ici :{' '}
              <Link to="/docs/developer/getting-started">Documentation Développeur</Link>.
            </p>
          </div>
        </section>
      </main>
    </Layout>
  );
}
