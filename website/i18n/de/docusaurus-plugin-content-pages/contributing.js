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
  'Deine erste Gravur',
  'Rotary-Einrichtung',
  'KI-Werkstückgenerator',
  'Kamera-Kalibrierung',
  'Print-&-Cut-Workflow',
  'Materialtests',
];

const quickActions = [
  {
    title: 'Fehler melden',
    description: 'Öffne ein Issue mit Schritten zum Reproduzieren und dem erwarteten Verhalten.',
    href: 'https://github.com/barebaric/rayforge/issues/new',
    icon: mdiBugOutline,
    iconClass: styles.iconCyan,
  },
  {
    title: 'Funktion vorschlagen',
    description: 'Teile einen Anwendungsfall und wie Erfolg für dich aussieht.',
    href: 'https://github.com/barebaric/rayforge/issues/new?labels=enhancement',
    icon: mdiLightbulbOnOutline,
    iconClass: styles.iconOrange,
  },
  {
    title: 'Code einreichen',
    description: 'Folge dem Entwickler-Leitfaden und sende einen Pull Request.',
    to: '/docs/developer/getting-started',
    icon: mdiSourcePull,
    iconClass: styles.iconPurple,
  },
  {
    title: 'Dokumentation verbessern',
    description: 'Korrigiere Tippfehler, füge Beispiele hinzu und mache die Doku leichter verständlich.',
    to: '/docs/getting-started/installation',
    icon: mdiBookOpenPageVariantOutline,
    iconClass: styles.iconCyan,
  },
  {
    title: 'Video-Tutorials erstellen',
    description:
      'Bringe der Welt Rayforge bei — fertige Tutorials werden auf der Startseite vorgestellt.',
    href: '#video-tutorials',
    icon: mdiVideoOutline,
    iconClass: styles.iconRed,
    featured: true,
  },
];

export default function Contributing() {
  return (
    <Layout
      title="Mitwirken"
      description="Erfahre, wie du bei Rayforge mitwirken kannst — melde Fehler, schlage Funktionen vor, reiche Code ein, erstelle Video-Tutorials, verbessere die Doku oder unterstütze das Projekt finanziell."
    >
      <main className={styles.pageWrapper}>
        <section className={styles.hero}>
          <div className={styles.heroInner}>
            <div className={styles.heroContent}>
              <h1 className={styles.heroTitle}>
                Bei{' '}
                <span className={styles.heroTitleGradient}>Rayforge</span>{' '}
                mitwirken
              </h1>
              <p className={styles.heroSubtitle}>
                Hilf mit, Rayforge zu verbessern: Melde Fehler, schlage
                Funktionen vor, reiche Code ein, verbessere die Doku, erstelle
                Tutorials oder unterstütze das Projekt finanziell.
              </p>
              <div className={styles.heroCtas}>
                <a
                  href="https://www.patreon.com/c/knipknap"
                  className={`rfButton rfButtonOrange ${styles.heroCtaButton}`}
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  <Icon path={mdiHandCoinOutline} size={0.9} />
                  <span>Auf Patreon unterstützen</span>
                </a>
                <a
                  href="https://github.com/barebaric/rayforge/issues/new"
                  className={`rfButton rfButtonDownload ${styles.heroCtaButton}`}
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  <Icon path={mdiBugOutline} size={0.9} />
                  <span>Fehler melden</span>
                </a>
                <Link
                  to="/docs/developer/getting-started"
                  className={`rfButton rfButtonPurple ${styles.heroCtaButton}`}
                >
                  <Icon path={mdiSourcePull} size={0.9} />
                  <span>Jetzt mitwirken</span>
                </Link>
              </div>
            </div>

            <div className={styles.heroPanel}>
              <div className={styles.panelHeader}>
                <div className={styles.panelBadge}>
                  <Icon path={mdiGithub} size={0.85} />
                  <span>GitHub</span>
                </div>
                <h2 className={styles.panelTitle}>Community & Unterstützung</h2>
              </div>
              <div className={styles.panelLinks}>
                <a
                  href="https://github.com/barebaric/rayforge/issues"
                  className={styles.panelLink}
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  <span className={styles.panelLinkLabel}>Probleme melden</span>
                  <span className={styles.panelLinkMeta}>GitHub Issues</span>
                </a>
                <a
                  href="https://github.com/barebaric/rayforge"
                  className={styles.panelLink}
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  <span className={styles.panelLinkLabel}>Quellcode ansehen</span>
                  <span className={styles.panelLinkMeta}>GitHub Repository</span>
                </a>
                <Link to="/sponsor" className={styles.panelLink}>
                  <span className={styles.panelLinkLabel}>
                    Sponsor werden
                  </span>
                  <span className={styles.panelLinkMeta}>Hilf uns, besser zu werden</span>
                </Link>
              </div>
            </div>
          </div>
        </section>

        <section className={styles.section}>
          <div className={styles.sectionInner}>
            <h2 className={styles.sectionTitle}>Den größten Unterschied machen</h2>
            <p className={styles.lead}>
              Manche Beiträge zählen mehr als andere. Gerade jetzt hilft
              Rayforge nichts mehr beim Wachsen als Video-Tutorials — und deine
              Großzügigkeit hält das Projekt am Leben.
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
                  <span>Sehr gefragt</span>
                </div>
                <div className={styles.impactCardHeader}>
                  <div className={`${styles.blockIcon} ${styles.iconRed}`}>
                    <Icon path={mdiYoutube} size={1.1} />
                  </div>
                  <h3 className={styles.impactCardTitle}>
                    Video-Tutorials erstellen
                  </h3>
                </div>
                <p className={styles.impactCardBody}>
                  Über Videos entdecken die meisten Menschen Rayforge — und
                  fertige Tutorials werden auf der Startseite mit deinem Namen
                  und einem Link zu deinem Kanal vorgestellt.
                </p>
                <ol className={styles.steps}>
                  <li className={styles.step}>
                    Wähle unten ein Thema — oder eines deiner eigenen.
                  </li>
                  <li className={styles.step}>
                    Nimm eine kurze Bildschirmaufnahme mit Kommentar auf, lade
                    sie auf YouTube hoch und teile den Link auf{' '}
                    <a
                      href="https://discord.gg/sTHNdTtpQJ"
                      target="_blank"
                      rel="noopener noreferrer"
                    >
                      Discord
                    </a>{' '}
                    oder in den{' '}
                    <a
                      href="https://github.com/barebaric/rayforge/discussions"
                      target="_blank"
                      rel="noopener noreferrer"
                    >
                      GitHub-Diskussionen
                    </a>{' '}
                    , um deinen Platz zu beanspruchen.
                  </li>
                </ol>
                <div className={styles.wishlist}>
                  <div className={styles.wishlistTitle}>
                    <Icon path={mdiFire} size={0.85} />
                    <span>Wunschliste — Thema sichern</span>
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
                  <span>Teile dein Tutorial</span>
                </a>
              </div>

              <div
                className={`${styles.impactCard} ${styles.impactSupport}`}
              >
                <div
                  className={`${styles.impactBadge} ${styles.impactBadgeSupport}`}
                >
                  <Icon path={mdiHandCoinOutline} size={0.8} />
                  <span>Hält das Projekt am Leben</span>
                </div>
                <div className={styles.impactCardHeader}>
                  <div className={`${styles.blockIcon} ${styles.iconPurple}`}>
                    <Icon path={mdiHandCoinOutline} size={1.1} />
                  </div>
                  <h3 className={styles.impactCardTitle}>
                    Finanziell unterstützen
                  </h3>
                </div>
                <p className={styles.impactCardBody}>
                  Rayforge ist kostenlos und bleibt kostenlos. Patreon- und
                  Sponsoring-Gelder finanzieren Server, Testhardware und
                  Entwicklungszeit — sie halten das Projekt in Bewegung.
                </p>
                <div className={styles.impactLinks}>
                  <a
                    href="https://www.patreon.com/c/knipknap"
                    className={`rfButton rfButtonOrange ${styles.impactCta}`}
                    target="_blank"
                    rel="noopener noreferrer"
                  >
                    <Icon path={mdiHandCoinOutline} size={0.9} />
                    <span>Auf Patreon unterstützen</span>
                  </a>
                  <Link
                    to="/sponsor"
                    className={`rfButton rfButtonPurple ${styles.impactCta}`}
                  >
                    <Icon path={mdiStarOutline} size={0.9} />
                    <span>Sponsor werden</span>
                  </Link>
                </div>
              </div>
            </div>
          </div>
        </section>

        <section className={styles.section}>
          <div className={styles.sectionInner}>
            <h2 className={styles.sectionTitle}>Schnellaktionen</h2>
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
                        <span>Vorgestellt werden</span>
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
            <h2 className={styles.sectionTitle}>Möglichkeiten mitzuwirken</h2>
            <p className={styles.lead}>
              Wir freuen uns über Beiträge jeder Art. Jeder Fehlerbericht,
              jedes PR und jede Doku-Korrektur macht Rayforge für alle besser.
            </p>

            <div className={styles.twoCol}>
              <div className={styles.block}>
                <div className={styles.blockHeader}>
                  <div className={`${styles.blockIcon} ${styles.iconCyan}`}>
                    <Icon path={mdiBugOutline} size={0.95} />
                  </div>
                  <h3 className={styles.blockTitle}>Fehler melden</h3>
                </div>
                <ol className={styles.steps}>
                  <li className={styles.step}>
                    Prüfe{' '}
                    <a
                      href="https://github.com/barebaric/rayforge/issues"
                      target="_blank"
                      rel="noopener noreferrer"
                    >
                      bestehende Issues
                    </a>{' '}
                    , um Duplikate zu vermeiden.
                  </li>
                  <li className={styles.step}>
                    Erstelle ein{' '}
                    <a
                      href="https://github.com/barebaric/rayforge/issues/new"
                      target="_blank"
                      rel="noopener noreferrer"
                    >
                      neues Issue
                    </a>{' '}
                    mit einer klaren Beschreibung, Schritten zum
                    Reproduzieren, erwartetem vs. tatsächlichem Verhalten,
                    Systeminformationen und Screenshots, falls zutreffend.
                  </li>
                </ol>
              </div>

              <div className={styles.block}>
                <div className={styles.blockHeader}>
                  <div className={`${styles.blockIcon} ${styles.iconOrange}`}>
                    <Icon path={mdiLightbulbOnOutline} size={0.95} />
                  </div>
                  <h3 className={styles.blockTitle}>Funktionen vorschlagen</h3>
                </div>
                <ol className={styles.steps}>
                  <li className={styles.step}>
                    Prüfe{' '}
                    <a
                      href="https://github.com/barebaric/rayforge/issues?q=is%3Aissue+label%3Aenhancement"
                      target="_blank"
                      rel="noopener noreferrer"
                    >
                      bestehende Funktionswünsche
                    </a>
                    .
                  </li>
                  <li className={styles.step}>
                    Eröffne einen Funktionswunsch mit der Idee, dem
                    Anwendungsfall, den Vorteilen und (optional) einem
                    möglichen Ansatz.
                  </li>
                </ol>
              </div>

              <div className={styles.block}>
                <div className={styles.blockHeader}>
                  <div className={`${styles.blockIcon} ${styles.iconPurple}`}>
                    <Icon path={mdiSourcePull} size={0.95} />
                  </div>
                  <h3 className={styles.blockTitle}>Code einreichen</h3>
                </div>
                <p className={styles.blockBody}>
                  Ausführliche Informationen zum Einreichen von Code-Beiträgen
                  findest du im Leitfaden{' '}
                  <Link to="/docs/developer/getting-started">
                    Entwickler-Dokumentation – Erste Schritte
                  </Link>{' '}
                  .
                </p>
              </div>

              <div className={styles.block}>
                <div className={styles.blockHeader}>
                  <div className={`${styles.blockIcon} ${styles.iconCyan}`}>
                    <Icon path={mdiBookOpenPageVariantOutline} size={0.95} />
                  </div>
                  <h3 className={styles.blockTitle}>Dokumentation verbessern</h3>
                </div>
                <ul className={styles.bullets}>
                  <li>Tippfehler oder unklare Erklärungen korrigieren</li>
                  <li>Beispiele und Screenshots hinzufügen</li>
                  <li>Struktur verbessern</li>
                  <li>In andere Sprachen übersetzen</li>
                </ul>
                <p className={styles.blockBody}>
                  Du kannst auf jeder Dokumentationsseite auf die Schaltfläche
                  „Diese Seite bearbeiten" klicken und dann PRs auf die gleiche
                  Weise wie Code-Beiträge einreichen.
                </p>
              </div>
            </div>
          </div>
        </section>

        <section className={styles.section}>
          <div className={styles.sectionInner}>
            <h2 className={styles.sectionTitle}>Über diese Dokumentation</h2>
            <p className={styles.lead}>
              Diese Dokumentation ist für Endanwender von Rayforge konzipiert.
              Für Entwickler-Dokumentation beginne hier:{' '}
              <Link to="/docs/developer/getting-started">Entwickler-Dokumentation</Link>.
            </p>
          </div>
        </section>
      </main>
    </Layout>
  );
}
