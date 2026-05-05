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
} from '@mdi/js';
import styles from './contributing.module.css';

const quickActions = [
  {
    title: 'Report a Bug',
    description: 'Open an issue with steps to reproduce and what you expected.',
    href: 'https://github.com/barebaric/rayforge/issues/new',
    icon: mdiBugOutline,
    iconClass: styles.iconCyan,
  },
  {
    title: 'Suggest a Feature',
    description: 'Share a use case and what success looks like for you.',
    href: 'https://github.com/barebaric/rayforge/issues/new?labels=enhancement',
    icon: mdiLightbulbOnOutline,
    iconClass: styles.iconOrange,
  },
  {
    title: 'Submit Code',
    description: 'Follow the developer guide and send a pull request.',
    to: '/docs/developer/getting-started',
    icon: mdiSourcePull,
    iconClass: styles.iconPurple,
  },
  {
    title: 'Improve Documentation',
    description: 'Fix typos, add examples, and make the docs easier to follow.',
    to: '/docs/getting-started/installation',
    icon: mdiBookOpenPageVariantOutline,
    iconClass: styles.iconCyan,
  },
];

export default function Contributing() {
  return (
    <Layout
      title="Contributing"
      description="Learn how to contribute to Rayforge — report bugs, suggest features, submit code, improve docs, or support the project financially."
    >
      <main className={styles.pageWrapper}>
        <section className={styles.hero}>
          <div className={styles.heroInner}>
            <div className={styles.heroContent}>
              <h1 className={styles.heroTitle}>
                Contributing to{' '}
                <span className={styles.heroTitleGradient}>Rayforge</span>
              </h1>
              <p className={styles.heroSubtitle}>
                Help improve Rayforge: report bugs, suggest features, submit
                code, refine docs, or support the project financially.
              </p>
              <div className={styles.heroCtas}>
                <a
                  href="https://www.patreon.com/c/knipknap"
                  className={`rfButton rfButtonOrange ${styles.heroCtaButton}`}
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  <Icon path={mdiHandCoinOutline} size={0.9} />
                  <span>Support on Patreon</span>
                </a>
                <a
                  href="https://github.com/barebaric/rayforge/issues/new"
                  className={`rfButton rfButtonDownload ${styles.heroCtaButton}`}
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  <Icon path={mdiBugOutline} size={0.9} />
                  <span>Report a Bug</span>
                </a>
                <Link
                  to="/docs/developer/getting-started"
                  className={`rfButton rfButtonPurple ${styles.heroCtaButton}`}
                >
                  <Icon path={mdiSourcePull} size={0.9} />
                  <span>Start Contributing</span>
                </Link>
              </div>
            </div>

            <div className={styles.heroPanel}>
              <div className={styles.panelHeader}>
                <div className={styles.panelBadge}>
                  <Icon path={mdiGithub} size={0.85} />
                  <span>GitHub</span>
                </div>
                <h2 className={styles.panelTitle}>Community & Support</h2>
              </div>
              <div className={styles.panelLinks}>
                <a
                  href="https://github.com/barebaric/rayforge/issues"
                  className={styles.panelLink}
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  <span className={styles.panelLinkLabel}>Report issues</span>
                  <span className={styles.panelLinkMeta}>GitHub Issues</span>
                </a>
                <a
                  href="https://github.com/barebaric/rayforge"
                  className={styles.panelLink}
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  <span className={styles.panelLinkLabel}>Browse source</span>
                  <span className={styles.panelLinkMeta}>GitHub repository</span>
                </a>
                <Link to="/sponsor" className={styles.panelLink}>
                  <span className={styles.panelLinkLabel}>
                    Become a Sponsor
                  </span>
                  <span className={styles.panelLinkMeta}>Help us Improve</span>
                </Link>
              </div>
            </div>
          </div>
        </section>

        <section className={styles.section}>
          <div className={styles.sectionInner}>
            <h2 className={styles.sectionTitle}>Quick Actions</h2>
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
            <h2 className={styles.sectionTitle}>Ways to Contribute</h2>
            <p className={styles.lead}>
              We welcome contributions of all kinds. Every bug report, PR, and
              documentation fix makes Rayforge better for everyone.
            </p>

            <div className={styles.twoCol}>
              <div className={styles.block}>
                <div className={styles.blockHeader}>
                  <div className={`${styles.blockIcon} ${styles.iconCyan}`}>
                    <Icon path={mdiBugOutline} size={0.95} />
                  </div>
                  <h3 className={styles.blockTitle}>Report Bugs</h3>
                </div>
                <ol className={styles.steps}>
                  <li className={styles.step}>
                    Check{' '}
                    <a
                      href="https://github.com/barebaric/rayforge/issues"
                      target="_blank"
                      rel="noopener noreferrer"
                    >
                      existing issues
                    </a>{' '}
                    to avoid duplicates.
                  </li>
                  <li className={styles.step}>
                    Create a{' '}
                    <a
                      href="https://github.com/barebaric/rayforge/issues/new"
                      target="_blank"
                      rel="noopener noreferrer"
                    >
                      new issue
                    </a>{' '}
                    with a clear description, steps to reproduce, expected vs.
                    actual behavior, system info, and screenshots if applicable.
                  </li>
                </ol>
              </div>

              <div className={styles.block}>
                <div className={styles.blockHeader}>
                  <div className={`${styles.blockIcon} ${styles.iconOrange}`}>
                    <Icon path={mdiLightbulbOnOutline} size={0.95} />
                  </div>
                  <h3 className={styles.blockTitle}>Suggest Features</h3>
                </div>
                <ol className={styles.steps}>
                  <li className={styles.step}>
                    Review{' '}
                    <a
                      href="https://github.com/barebaric/rayforge/issues?q=is%3Aissue+label%3Aenhancement"
                      target="_blank"
                      rel="noopener noreferrer"
                    >
                      existing feature requests
                    </a>
                    .
                  </li>
                  <li className={styles.step}>
                    Open a feature request describing the idea, use case,
                    benefits, and (optionally) a possible approach.
                  </li>
                </ol>
              </div>

              <div className={styles.block}>
                <div className={styles.blockHeader}>
                  <div className={`${styles.blockIcon} ${styles.iconPurple}`}>
                    <Icon path={mdiSourcePull} size={0.95} />
                  </div>
                  <h3 className={styles.blockTitle}>Submit Code</h3>
                </div>
                <p className={styles.blockBody}>
                  For detailed information on submitting code contributions,
                  follow the{' '}
                  <Link to="/docs/developer/getting-started">
                    Developer Documentation – Getting Started
                  </Link>{' '}
                  guide.
                </p>
              </div>

              <div className={styles.block}>
                <div className={styles.blockHeader}>
                  <div className={`${styles.blockIcon} ${styles.iconCyan}`}>
                    <Icon path={mdiBookOpenPageVariantOutline} size={0.95} />
                  </div>
                  <h3 className={styles.blockTitle}>Improve Documentation</h3>
                </div>
                <ul className={styles.bullets}>
                  <li>Fix typos or unclear explanations</li>
                  <li>Add examples and screenshots</li>
                  <li>Improve organization</li>
                  <li>Translate to other languages</li>
                </ul>
                <p className={styles.blockBody}>
                  Use the “edit this page” button on any documentation page and
                  submit PRs the same way as code contributions.
                </p>
              </div>
            </div>
          </div>
        </section>

        <section className={styles.section}>
          <div className={styles.sectionInner}>
            <h2 className={styles.sectionTitle}>About This Documentation</h2>
            <p className={styles.lead}>
              This documentation is designed for end-users of Rayforge. For
              developer docs, start here:{' '}
              <Link to="/docs/developer/getting-started">Developer Documentation</Link>.
            </p>
          </div>
        </section>
      </main>
    </Layout>
  );
}
