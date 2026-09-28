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
  'रोटरी उत्कीर्णन सेटअप',
  'AI वर्कपीस जनरेटर',
  'कैमरा कैलिब्रेशन',
  'प्रिंट और कट वर्कफ़्लो',
  'सामग्री परीक्षण',
];

const quickActions = [
  {
    title: 'बग रिपोर्ट करें',
    description:
      'पुनरुत्पादन के चरणों और अपेक्षित परिणाम के साथ एक समस्या खोलें.',
    href: 'https://github.com/barebaric/rayforge/issues/new',
    icon: mdiBugOutline,
    iconClass: styles.iconCyan,
  },
  {
    title: 'सुविधा सुझाएँ',
    description: 'अपना उपयोग प्रसंग और सफलता कैसी दिखे, साझा करें.',
    href: 'https://github.com/barebaric/rayforge/issues/new?labels=enhancement',
    icon: mdiLightbulbOnOutline,
    iconClass: styles.iconOrange,
  },
  {
    title: 'कोड सबमिट करें',
    description: 'डेवलपर गाइड का पालन करें और एक पुल अनुरोध भेजें.',
    to: '/docs/developer/getting-started',
    icon: mdiSourcePull,
    iconClass: styles.iconPurple,
  },
  {
    title: 'दस्तावेज़ीकरण सुधारें',
    description:
      'टाइपो ठीक करें, उदाहरण जोड़ें, और दस्तावेज़ को अधिक सुगम बनाएँ.',
    to: '/docs/getting-started/installation',
    icon: mdiBookOpenPageVariantOutline,
    iconClass: styles.iconCyan,
  },
];

export default function Contributing() {
  return (
    <Layout
      title="Contributing"
      description="Learn how to contribute to Rayforge — report bugs, suggest features, submit code, create video tutorials, improve docs, or support the project financially."
    >
      <main className={styles.pageWrapper}>
        <section className={styles.hero}>
          <div className={styles.heroInner}>
            <div className={styles.heroContent}>
              <h1 className={styles.heroTitle}>
                <span className={styles.heroTitleGradient}>Rayforge</span>{' '}
                में योगदान
              </h1>
              <p className={styles.heroSubtitle}>
                Rayforge को बेहतर बनाने में मदद करें: बग रिपोर्ट करें,
                सुविधाएँ सुझाएँ, कोड सबमिट करें, दस्तावेज़ सुधारें,
                ट्यूटोरियल बनाएँ, या परियोजना को वित्तीय रूप से समर्थन दें.
              </p>
              <div className={styles.heroCtas}>
                <a
                  href="https://www.patreon.com/c/knipknap"
                  className={`rfButton rfButtonOrange ${styles.heroCtaButton}`}
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  <Icon path={mdiHandCoinOutline} size={0.9} />
                  <span>Patreon पर समर्थन करें</span>
                </a>
                <a
                  href="https://github.com/barebaric/rayforge/issues/new"
                  className={`rfButton rfButtonDownload ${styles.heroCtaButton}`}
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  <Icon path={mdiBugOutline} size={0.9} />
                  <span>बग रिपोर्ट करें</span>
                </a>
                <Link
                  to="/docs/developer/getting-started"
                  className={`rfButton rfButtonPurple ${styles.heroCtaButton}`}
                >
                  <Icon path={mdiSourcePull} size={0.9} />
                  <span>योगदान शुरू करें</span>
                </Link>
              </div>
            </div>

            <div className={styles.heroPanel}>
              <div className={styles.panelHeader}>
                <div className={styles.panelBadge}>
                  <Icon path={mdiGithub} size={0.85} />
                  <span>GitHub</span>
                </div>
                <h2 className={styles.panelTitle}>समुदाय और समर्थन</h2>
              </div>
              <div className={styles.panelLinks}>
                <a
                  href="https://github.com/barebaric/rayforge/issues"
                  className={styles.panelLink}
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  <span className={styles.panelLinkLabel}>समस्याएँ रिपोर्ट करें</span>
                  <span className={styles.panelLinkMeta}>GitHub Issues</span>
                </a>
                <a
                  href="https://github.com/barebaric/rayforge"
                  className={styles.panelLink}
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  <span className={styles.panelLinkLabel}>स्रोत देखें</span>
                  <span className={styles.panelLinkMeta}>GitHub रिपॉज़िटरी</span>
                </a>
                <Link to="/sponsor" className={styles.panelLink}>
                  <span className={styles.panelLinkLabel}>
                    प्रायोजक बनें
                  </span>
                  <span className={styles.panelLinkMeta}>सुधार में मदद करें</span>
                </Link>
              </div>
            </div>
          </div>
        </section>

        <section className={styles.section}>
          <div className={styles.sectionInner}>
            <h2 className={styles.sectionTitle}>सबसे बड़ा प्रभाव डालें</h2>
            <p className={styles.lead}>
              कुछ योगदान अन्य की तुलना में अधिक बदलाव लाते हैं. इस समय,
              वीडियो ट्यूटोरियल जितना Rayforge को बढ़ाने में कुछ भी मदद नहीं
              करता — और आपकी उदारता परियोजना को जीवित रखती है.
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
                  <span>सर्वाधिक वांछित</span>
                </div>
                <div className={styles.impactCardHeader}>
                  <div className={`${styles.blockIcon} ${styles.iconRed}`}>
                    <Icon path={mdiYoutube} size={1.1} />
                  </div>
                  <h3 className={styles.impactCardTitle}>
                    वीडियो ट्यूटोरियल बनाएँ
                  </h3>
                </div>
                <p className={styles.impactCardBody}>
                  अधिकांश लोग Rayforge को वीडियो से खोजते हैं — और पूरे किए
                  गए ट्यूटोरियल आपके नाम और चैनल लिंक के साथ मुख्य पृष्ठ पर
                  प्रदर्शित होते हैं.
                </p>
                <ol className={styles.steps}>
                  <li className={styles.step}>
                    नीचे से एक विषय चुनें — या अपना स्वयं का चुनें.
                  </li>
                  <li className={styles.step}>
                    वॉयसओवर के साथ एक छोटा स्क्रीन कैप्चर रिकॉर्ड करें, इसे
                    YouTube पर अपलोड करें, और अपना स्थान पाने के लिए लिंक{' '}
                    <a
                      href="https://discord.gg/sTHNdTtpQJ"
                      target="_blank"
                      rel="noopener noreferrer"
                    >
                      Discord
                    </a>{' '}
                    या{' '}
                    <a
                      href="https://github.com/barebaric/rayforge/discussions"
                      target="_blank"
                      rel="noopener noreferrer"
                    >
                      GitHub Discussions
                    </a>{' '}
                    पर साझा करें.
                  </li>
                </ol>
                <div className={styles.wishlist}>
                  <div className={styles.wishlistTitle}>
                    <Icon path={mdiFire} size={0.85} />
                    <span>विशलिस्ट — एक विषय चुनें</span>
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
                  <span>अपना ट्यूटोरियल साझा करें</span>
                </a>
              </div>

              <div
                className={`${styles.impactCard} ${styles.impactSupport}`}
              >
                <div
                  className={`${styles.impactBadge} ${styles.impactBadgeSupport}`}
                >
                  <Icon path={mdiHandCoinOutline} size={0.8} />
                  <span>परियोजना को जीवित रखता है</span>
                </div>
                <div className={styles.impactCardHeader}>
                  <div className={`${styles.blockIcon} ${styles.iconPurple}`}>
                    <Icon path={mdiHandCoinOutline} size={1.1} />
                  </div>
                  <h3 className={styles.impactCardTitle}>
                    वित्तीय समर्थन दें
                  </h3>
                </div>
                <p className={styles.impactCardBody}>
                  Rayforge मुफ़्त है, और यह मुफ़्त ही रहेगा. Patreon और
                  प्रायोजन धन सर्वर, परीक्षण हार्डवेयर, और विकास समय का भुगतान
                  करता है — यह परियोजना को आगे बढ़ाए रखता है.
                </p>
                <div className={styles.impactLinks}>
                  <a
                    href="https://www.patreon.com/c/knipknap"
                    className={`rfButton rfButtonOrange ${styles.impactCta}`}
                    target="_blank"
                    rel="noopener noreferrer"
                  >
                    <Icon path={mdiHandCoinOutline} size={0.9} />
                    <span>Patreon पर समर्थन करें</span>
                  </a>
                  <Link
                    to="/sponsor"
                    className={`rfButton rfButtonPurple ${styles.impactCta}`}
                  >
                    <Icon path={mdiStarOutline} size={0.9} />
                    <span>प्रायोजक बनें</span>
                  </Link>
                </div>
              </div>
            </div>
          </div>
        </section>

        <section className={styles.section}>
          <div className={styles.sectionInner}>
            <h2 className={styles.sectionTitle}>त्वरित क्रियाएँ</h2>
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
            <h2 className={styles.sectionTitle}>इस दस्तावेज़ीकरण के बारे में</h2>
            <p className={styles.lead}>
              यह दस्तावेज़ीकरण Rayforge के अंतिम उपयोगकर्ताओं के लिए है.
              डेवलपर दस्तावेज़ के लिए यहाँ शुरू करें:{' '}
              <Link to="/docs/developer/getting-started">डेवलपर दस्तावेज़ीकरण</Link>.
            </p>
          </div>
        </section>
      </main>
    </Layout>
  );
}
