import React from 'react';
import Layout from '@theme/Layout';
import Icon from '@mdi/react';
import { translate } from '@docusaurus/Translate';
import {
  mdiRocketLaunch,
  mdiMonitor,
  mdiStarOutline,
  mdiWrench,
  mdiYoutube,
  mdiGithub,
  mdiEmailOutline,
  mdiPalette,
  mdiFlaskOutline,
  mdiChartLine,
  mdiShieldCheck,
  mdiHandshake,
  mdiPlayCircleOutline,
  mdiBookmarkOutline,
  mdiCodeTags,
  mdiAccountGroup,
} from '@mdi/js';
import styles from './sponsor.module.css';

const benefits = [
  {
    icon: mdiRocketLaunch,
    iconClass: styles.benefitIconCyan,
    title: translate({
      id: 'sponsor.benefit.brandSelector.title',
      message: 'First-Run Setup',
    }),
    description: translate({
      id: 'sponsor.benefit.brandSelector.description',
      message:
        "When someone launches Rayforge for the first time, they pick their brand and model - and everything just works automatically.",
    }),
  },
  {
    icon: mdiMonitor,
    iconClass: styles.benefitIconPurple,
    title: translate({
      id: 'sponsor.benefit.machineProfile.title',
      message: 'Branded Profiles',
    }),
    description: translate({
      id: 'sponsor.benefit.machineProfile.description',
      message:
        "Your machines show up with your logo and photos right in the selector. Your brand, front and center, every day.",
    }),
  },
  {
    icon: mdiStarOutline,
    iconClass: styles.benefitIconOrange,
    title: translate({
      id: 'sponsor.benefit.showcase.title',
      message: 'Recommended Showcase',
    }),
    description: translate({
      id: 'sponsor.benefit.showcase.description',
      message:
        'A dedicated spot in the app and on the website showing off partner machines with photos, specs, and purchase links.',
    }),
  },
  {
    icon: mdiShieldCheck,
    iconClass: styles.benefitIconCyan,
    title: translate({
      id: 'sponsor.benefit.badgeProgram.title',
      message: '"Works with Rayforge"',
    }),
    description: translate({
      id: 'sponsor.benefit.badgeProgram.description',
      message:
        'Receive branded badge graphics for your website, packaging, and marketplace listings, which link back to Rayforge to build higher customer trust.',
    }),
  },
  {
    icon: mdiFlaskOutline,
    iconClass: styles.benefitIconPurple,
    title: translate({
      id: 'sponsor.benefit.materialPresets.title',
      message: 'Material Presets',
    }),
    description: translate({
      id: 'sponsor.benefit.materialPresets.description',
      message:
        'We dial in power, speed, and pass settings specifically for your machines. Fewer support headaches and happier customers.',
    }),
  },
  {
    icon: mdiPalette,
    iconClass: styles.benefitIconOrange,
    title: translate({
      id: 'sponsor.benefit.coBrandedInstall.title',
      message: 'Bundled Install',
    }),
    description: translate({
      id: 'sponsor.benefit.coBrandedInstall.description',
      message:
        "Ship Rayforge with your machine via QR code or a custom installer. Save the cost of building your own software.",
    }),
  },
  {
    icon: mdiWrench,
    iconClass: styles.benefitIconCyan,
    title: translate({
      id: 'sponsor.benefit.prioritySupport.title',
      message: 'Priority Support',
    }),
    description: translate({
      id: 'sponsor.benefit.prioritySupport.description',
      message:
        "Got a new controller? We'll add support for it fast-think release day, not months later. Your roadmap is covered.",
    }),
  },
  {
    icon: mdiYoutube,
    iconClass: styles.benefitIconRed,
    title: translate({
      id: 'sponsor.benefit.youtubeSpotlights.title',
      message: 'Machine Spotlights',
    }),
    description: translate({
      id: 'sponsor.benefit.youtubeSpotlights.description',
      message:
        "Tutorials, Setup, Unboxing, Material Tests, and more - all on the Barebaric YouTube channel.",
    }),
  },
  {
    icon: mdiGithub,
    iconClass: styles.benefitIconPurple,
    title: translate({
      id: 'sponsor.benefit.githubVisibility.title',
      message: 'GitHub Visibility',
    }),
    description: translate({
      id: 'sponsor.benefit.githubVisibility.description',
      message:
        'Your logo in the Rayforge README seen by developers and makers. Machine profiles get committed with "Contributed by Brand".',
    }),
  },
  {
    icon: mdiAccountGroup,
    iconClass: styles.benefitIconRed,
    title: translate({
      id: 'sponsor.benefit.forumPresence.title',
      message: 'Community Presence',
    }),
    description: translate({
      id: 'sponsor.benefit.forumPresence.description',
      message:
        'An official spot for your brand on GitHub Discussions and Discord to chat with users directly.',
    }),
  },
  {
    icon: mdiChartLine,
    iconClass: styles.benefitIconOrange,
    title: translate({
      id: 'sponsor.benefit.analytics.title',
      message: 'User Insights',
    }),
    description: translate({
      id: 'sponsor.benefit.analytics.description',
      message:
        "Aggregated, anonymized data on what materials and features your customers are actually using. (All opt-in, of course.)",
    }),
  },
  {
    icon: mdiHandshake,
    iconClass: styles.benefitIconPurple,
    title: translate({
      id: 'sponsor.benefit.coMarketing.title',
      message: 'Cross-Promotion',
    }),
    description: translate({
      id: 'sponsor.benefit.coMarketing.description',
      message:
        "Blog posts, social media shout-outs, and reviews across YouTube, GitHub, and Discord.",
    }),
  },
];

export default function Sponsor() {
  return (
    <Layout
      title={translate({
        id: 'sponsor.title',
        message: 'Sponsor Rayforge',
      })}
      description={translate({
        id: 'sponsor.description',
        message:
          "Get your laser machines into Rayforge — pre-configured, showcased, and backed by video content and tutorials. Let's work together!",
      })}
    >
      <main className={styles.pageWrapper}>
        {/* Hero */}
        <section className={styles.hero}>
          <div className={styles.heroInner}>
            <h1 className={styles.heroTitle}>
              <span className={styles.heroTitleGradient}>
                {translate({
                  id: 'sponsor.hero.gradient',
                  message: 'Partner',
                })}
              </span>{' '}
              {translate({
                id: 'sponsor.hero.title',
                message: 'with Rayforge',
              })}
            </h1>
            <p className={styles.heroSubtitle}>
              {translate({
                id: 'sponsor.hero.subtitle',
                message:
                  "Get your machines in front of laser cutting and engraving enthusiasts everywhere. We'll handle the software, the tutorials, and the community — you ship great hardware.",
              })}
            </p>
            <a
              href="mailto:sam@barebaric.com?subject=Sponsorship%20Inquiry"
              className={styles.heroCta}
            >
              <Icon path={mdiEmailOutline} size={0.85} />
              {translate({
                id: 'sponsor.hero.cta',
                message: 'Get in Touch',
              })}
            </a>
          </div>
        </section>

        {/* Stats */}
        <section className={styles.statsSection}>
          <div className={styles.statsGrid}>
            <div className={styles.statItem}>
              <div className={styles.statNumber}>750+</div>
              <div className={styles.statLabel}>
                {translate({
                  id: 'sponsor.stats.downloads',
                  message: 'Monthly Downloads',
                })}
              </div>
            </div>
            <div className={styles.statItem}>
              <div className={styles.statNumber}>179</div>
              <div className={styles.statLabel}>
                {translate({
                  id: 'sponsor.stats.stars',
                  message: 'GitHub Stars',
                })}
              </div>
            </div>
            <div className={styles.statItem}>
              <div className={styles.statNumber}>3,000</div>
              <div className={styles.statLabel}>
                {translate({
                  id: 'sponsor.stats.commits',
                  message: 'Commits',
                })}
              </div>
            </div>
            <div className={styles.statItem}>
              <div className={styles.statNumber}>6</div>
              <div className={styles.statLabel}>
                {translate({
                  id: 'sponsor.stats.languages',
                  message: 'Languages',
                })}
              </div>
            </div>
            <div className={styles.statItem}>
              <div className={styles.statNumber}>3</div>
              <div className={styles.statLabel}>
                {translate({
                  id: 'sponsor.stats.platforms',
                  message: 'Platforms',
                })}
              </div>
            </div>
            <div className={styles.statItem}>
              <div className={styles.statNumber}>100%</div>
              <div className={styles.statLabel}>
                {translate({
                  id: 'sponsor.stats.opensource',
                  message: 'Free & Open Source',
                })}
              </div>
            </div>
          </div>
        </section>

        {/* Benefits */}
        <section className={styles.benefitsSection}>
          <h2 className={styles.sectionTitle}>
            {translate({
              id: 'sponsor.benefits.title',
              message: 'What You Get',
            })}
          </h2>
          <p className={styles.sectionSubtitle}>
            {translate({
              id: 'sponsor.benefits.subtitle',
              message:
                "We're building real integrations, not just slapping your logo on a page. Here's what a partnership looks like.",
            })}
          </p>
          <div className={styles.benefitGrid}>
            {benefits.map((benefit, i) => (
              <div className={styles.benefitCard} key={i}>
                <div
                  className={`${styles.benefitIcon} ${benefit.iconClass}`}
                >
                  <Icon path={benefit.icon} size={1.3} />
                </div>
                <h3 className={styles.benefitTitle}>{benefit.title}</h3>
                <p className={styles.benefitDescription}>
                  {benefit.description}
                </p>
              </div>
            ))}
          </div>
        </section>

        {/* Comparison */}
        <section className={styles.comparisonSection}>
          <div className={styles.comparisonInner}>
            <h2 className={styles.sectionTitle}>
              {translate({
                id: 'sponsor.comparison.title',
                message: 'Why Rayforge?',
              })}
            </h2>
            <p className={styles.sectionSubtitle}>
              {translate({
                id: 'sponsor.comparison.subtitle',
                message:
                  "No other free laser software offers manufacturers anything like this. Here's how we compare.",
              })}
            </p>
            <table className={styles.comparisonTable}>
              <thead>
                <tr>
                  <th></th>
                  <th>LightBurn</th>
                  <th>LaserGRBL</th>
                  <th className={styles.comparisonHighlight}>
                    Rayforge
                  </th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td>
                    {translate({
                      id: 'sponsor.comparison.costToUser',
                      message: 'Cost to User',
                    })}
                  </td>
                  <td>$99–199</td>
                  <td>
                    {translate({
                      id: 'sponsor.comparison.free',
                      message: 'Free',
                    })}
                  </td>
                  <td className={styles.comparisonHighlight}>
                    {translate({
                      id: 'sponsor.comparison.free',
                      message: 'Free',
                    })}
                  </td>
                </tr>
                <tr>
                  <td>
                    {translate({
                      id: 'sponsor.comparison.costToBrand',
                      message: 'Cost to Brand',
                    })}
                  </td>
                  <td>
                    {translate({
                      id: 'sponsor.comparison.lb.costToBrand',
                      message: 'Per-license resale',
                    })}
                  </td>
                  <td>
                    {translate({
                      id: 'sponsor.comparison.grbl.costToBrand',
                      message: 'No program',
                    })}
                  </td>
                  <td className={styles.comparisonHighlight}>
                    {translate({
                      id: 'sponsor.comparison.rf.costToBrand',
                      message: 'Optional Partnership',
                    })}
                  </td>
                </tr>
                <tr>
                  <td>
                    {translate({
                      id: 'sponsor.comparison.brandVisibility',
                      message: 'Brand Visibility',
                    })}
                  </td>
                  <td>
                    {translate({
                      id: 'sponsor.comparison.lb.brandVisibility',
                      message: 'Compatibility list',
                    })}
                  </td>
                  <td>
                    {translate({
                      id: 'sponsor.comparison.grbl.brandVisibility',
                      message: 'Affiliate coupons',
                    })}
                  </td>
                  <td className={styles.comparisonHighlight}>
                    {translate({
                      id: 'sponsor.comparison.rf.brandVisibility',
                      message:
                        'In-app first-run selector + showcase',
                    })}
                  </td>
                </tr>
                <tr>
                  <td>
                    {translate({
                      id: 'sponsor.comparison.machineProfiles',
                      message: 'Machine Profiles',
                    })}
                  </td>
                  <td>
                    {translate({
                      id: 'sponsor.comparison.lb.machineProfiles',
                      message: 'Generic (by controller)',
                    })}
                  </td>
                  <td>
                    {translate({
                      id: 'sponsor.comparison.grbl.machineProfiles',
                      message: 'None',
                    })}
                  </td>
                  <td className={styles.comparisonHighlight}>
                    {translate({
                      id: 'sponsor.comparison.rf.machineProfiles',
                      message: 'Brand-specific, pre-tuned',
                    })}
                  </td>
                </tr>
                <tr>
                  <td>
                    {translate({
                      id: 'sponsor.comparison.linuxSupport',
                      message: 'Linux Support',
                    })}
                  </td>
                  <td>
                    {translate({
                      id: 'sponsor.comparison.no',
                      message: 'No',
                    })}
                  </td>
                  <td>
                    {translate({
                      id: 'sponsor.comparison.no',
                      message: 'No',
                    })}
                  </td>
                  <td className={styles.comparisonHighlight}>
                    {translate({
                      id: 'sponsor.comparison.rf.linuxSupport',
                      message: 'Native (GTK4 / Flatpak / Snap)',
                    })}
                  </td>
                </tr>
                <tr>
                  <td>
                    {translate({
                      id: 'sponsor.comparison.crossPlatform',
                      message: 'Cross-Platform',
                    })}
                  </td>
                  <td>Win + Mac</td>
                  <td>
                    {translate({
                      id: 'sponsor.comparison.grbl.crossPlatform',
                      message: 'Windows only',
                    })}
                  </td>
                  <td className={styles.comparisonHighlight}>
                    Win + Mac + Linux
                  </td>
                </tr>
                <tr>
                  <td>
                    {translate({
                      id: 'sponsor.comparison.devPace',
                      message: 'Development Pace',
                    })}
                  </td>
                  <td>
                    {translate({
                      id: 'sponsor.comparison.lb.devPace',
                      message: 'Moderate',
                    })}
                  </td>
                  <td>
                    {translate({
                      id: 'sponsor.comparison.grbl.devPace',
                      message: 'Stalling',
                    })}
                  </td>
                  <td className={styles.comparisonHighlight}>
                    {translate({
                      id: 'sponsor.comparison.rf.devPace',
                      message: 'Weekly releases',
                    })}
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
        </section>

        {/* CTA */}
        <section className={styles.ctaSection}>
          <div className={styles.ctaInner}>
            <h2 className={styles.ctaTitle}>
              {translate({
                id: 'sponsor.cta.title',
                message: "Let's Talk",
              })}
            </h2>
            <p className={styles.ctaText}>
              {translate({
                id: 'sponsor.cta.text',
                message:
                  "Whether you want your machines pre-configured in Rayforge, a dedicated video review, or something totally custom — drop me a line. I'm flexible and happy to work out something that makes sense for both of us. Hardware loans for profile development are always welcome too.",
              })}
            </p>
            <a
              href="mailto:sam@barebaric.com?subject=Sponsorship%20Inquiry"
              className={styles.heroCta}
            >
              <Icon path={mdiEmailOutline} size={0.85} />
              {translate({
                id: 'sponsor.cta.button',
                message: 'Say Hello',
              })}
            </a>
          </div>
        </section>
      </main>
    </Layout>
  );
}
