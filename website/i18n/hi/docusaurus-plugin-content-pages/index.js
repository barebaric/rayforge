import React, { useEffect, useState } from 'react';
import Layout from '@theme/Layout';
import Link from '@docusaurus/Link';
import styles from '@site/src/pages/index.module.css';
import Icon from '@mdi/react';
import {
  mdiDownload,
  mdiGithub,
  mdiArrowRight,
  mdiPlayCircleOutline,
  mdiYoutube,
  mdiShareVariant,
  mdiVectorSquare,
  mdiLayersOutline,
  mdiCameraOutline,
  mdiRotate3d,
  mdiBookOpenOutline,
  mdiMapOutline,
} from '@mdi/js';
import { tutorials } from '@site/src/data/tutorials';
import { references } from '@site/src/data/references';

function detectOs() {
  if (typeof window === 'undefined') {
    return 'linux';
  }

  const userAgent = window.navigator.userAgent.toLowerCase();

  if (userAgent.includes('win')) {
    return 'windows';
  }
  if (
    userAgent.includes('mac') ||
    userAgent.includes('iphone') ||
    userAgent.includes('ipad')
  ) {
    return 'macos';
  }
  if (userAgent.includes('linux')) {
    return 'linux';
  }

  return 'linux';
}

function HeroSection() {
  const [os, setOs] = useState('linux');

  useEffect(() => {
    setOs(detectOs());
  }, []);

  const downloadTo = `/docs/getting-started/installation#${os}`;

  return (
    <section className={styles.hero}>
      <div className={styles.heroInner}>

        <div className={styles.heroContent}>
          <p className={styles.kicker}>डिज़ाइन / तैयारी / निर्माण</p>
          <h1 className={styles.heroTitle}>
            <span className={styles.heroTitleLine1}>विचारों से</span>
            <span className={styles.heroTitleLine2}>वास्तविक परियोजनाओं तक</span>
          </h1>
          <p className={styles.heroSubtitle}>
            Rayforge आपके लेज़र कटर के लिए क्रिएटिव सूट है. डिज़ाइन करें,
            तैयार करें और बनाएँ — सब एक ही मुफ़्त, ओपन सोर्स ऐप में.
          </p>
          <div className={styles.heroCtaButtons}>
            <Link to={downloadTo} className={styles.buttonDark}>
              <Icon path={mdiDownload} size={0.85} />
              <span>मुफ़्त में डाउनलोड करें</span>
            </Link>
            <a
              href="https://github.com/barebaric/rayforge"
              className={styles.buttonOutline}
              target="_blank"
              rel="noopener noreferrer"
            >
              <Icon path={mdiGithub} size={0.85} />
              <span>ओपन सोर्स</span>
            </a>
          </div>
          <a
            href="https://www.youtube.com/watch?v=srKXs2p31VY"
            className={styles.heroVideoLink}
            target="_blank"
            rel="noopener noreferrer"
          >
            <Icon path={mdiPlayCircleOutline} size={0.9} />
            <span>परिचय देखें</span>
            <Icon path={mdiArrowRight} size={0.65} />
          </a>
        </div>

      </div>
    </section>
  );
}

const capabilities = [
  {
    icon: mdiVectorSquare,
    label: '2D CAD स्केचर',
    to: '/docs/features/sketcher',
  },
  {
    icon: mdiLayersOutline,
    label: 'बहु-लेयर जॉब',
    to: '/docs/features/multi-layer',
  },
  {
    icon: mdiCameraOutline,
    label: 'कैमरा संरेखण',
    to: '/docs/machine/camera',
  },
  {
    icon: mdiRotate3d,
    label: 'रोटरी समर्थन',
    to: '/docs/machine/rotary',
  },
  {
    icon: mdiBookOpenOutline,
    label: 'सामग्री रेसिपी',
    to: '/docs/application-settings/recipes',
  },
  {
    icon: mdiMapOutline,
    label: 'पाथ अनुकूलन',
    to: '/docs/features/path-optimization',
  },
];

function CapabilityStrip() {
  return (
    <section className={styles.stripSection}>
      <div className={styles.stripInner}>
        {capabilities.map((cap) => (
          <Link key={cap.label} to={cap.to} className={styles.stripItem}>
            <Icon path={cap.icon} size={1.15} />
            <span>{cap.label}</span>
          </Link>
        ))}
      </div>
    </section>
  );
}

function DesignYourPartsSection() {
  return (
    <section className={styles.partsSection}>
      <div className={styles.partsLayers}>
        <div className={styles.partsLeft} />
        <div className={styles.partsRight} />
      </div>
      <div className={styles.partsInner}>
        <div className={styles.partsContent}>
          <p className={styles.partsKicker}>
            शक्तिशाली उपकरण. असीमित संभावनाएँ.
          </p>
          <h2 className={styles.partsTitle}>अपने पार्ट स्वयं बनाएँ</h2>
          <p className={styles.partsText}>
            कस्टम डिज़ाइन को सीधे Rayforge के भीतर स्केच, आकार दें और परिष्कृत
            करें. अंतर्निहित ड्रॉइंग उपकरण किसी भी विचार को जीवंत कर देते हैं —
            या बताएँ कि आप क्या चाहते हैं और AI वर्कपीस जनरेटर तुरंत आपके लिए
            डिज़ाइन कर देता है.
          </p>
          <Link to="/docs/features/sketcher" className={styles.partsLink}>
            <span>और जानें</span>
            <Icon path={mdiArrowRight} size={0.65} />
          </Link>
        </div>
      </div>
    </section>
  );
}

function FeatureCardsSection() {
  const cards = [
    {
      title: 'डिज़ाइन',
      subtitle: 'पैरामीट्रिक उपकरणों वाला शक्तिशाली 2D CAD स्केचर.',
      image: '/images/screenshot-sketcher.webp',
    },
    {
      title: 'तैयारी',
      subtitle:
        'छवियाँ ट्रेस करें, टूलपाथ अनुकूलित करें, और हर विवरण बारीकी से सुधारें.',
      image: '/images/screenshot-optimizer.webp',
    },
    {
      title: 'निर्माण',
      subtitle:
        'लेज़र और CNC जॉब पूर्ण विश्वास से चलाएँ. तेज़. सटीक. विश्वसनीय.',
      image: '/screenshots/main-3d-bee.webp',
    },
  ];

  return (
    <section className={styles.cardsSection}>
      <div className={styles.cardsTrees}>
        <div className={styles.cardsTreeLeft} />
        <div className={styles.cardsTreeRight} />
      </div>
      <div className={styles.cardsInner}>
        <p className={styles.cardsKicker}>जो आवश्यक है वह सब. जो नहीं वह कुछ नहीं.</p>
        <div className={styles.cardsGrid}>
          {cards.map((card) => (
            <div key={card.title} className={styles.card}>
              <div className={styles.cardImage}>
                <img src={card.image} alt={card.title} loading="lazy" />
              </div>
              <div className={styles.cardBody}>
                <h3 className={styles.cardTitle}>{card.title}</h3>
                <p className={styles.cardSubtitle}>{card.subtitle}</p>
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

function TutorialSpotlight() {
  return (
    <section className={styles.spotlightSection}>
      <div className={styles.spotlightInner}>
        <div className={styles.spotlightHeader}>
          <p className={styles.kicker}>समुदाय</p>
          <h2 className={styles.spotlightTitle}>वास्तविक उपयोगकर्ताओं के ट्यूटोरियल</h2>
          <p className={styles.spotlightSubtitle}>
            वास्तविक Rayforge उपयोगकर्ताओं के बनाए वीडियो से सीखें. आपका
            ट्यूटोरियल अगला यहाँ हो सकता है.
          </p>
        </div>

        {tutorials.length > 0 ? (
          <div className={styles.spotlightGrid}>
            {tutorials.map((tutorial) => (
              <a
                key={tutorial.id}
                href={`https://www.youtube.com/watch?v=${tutorial.id}`}
                target="_blank"
                rel="noopener noreferrer"
                className={styles.spotlightCard}
              >
                <div className={styles.spotlightThumb}>
                  <img
                    src={`https://img.youtube.com/vi/${tutorial.id}/hqdefault.jpg`}
                    alt={tutorial.title}
                    loading="lazy"
                  />
                  <span className={styles.spotlightPlay}>
                    <Icon path={mdiPlayCircleOutline} size={1.4} />
                  </span>
                </div>
                <h3 className={styles.spotlightVideoTitle}>{tutorial.title}</h3>
                <span className={styles.spotlightCreator}>
                  {tutorial.creator}
                </span>
              </a>
            ))}
          </div>
        ) : (
          <div className={styles.spotlightEmpty}>
            <div className={styles.spotlightEmptyIcon}>
              <Icon path={mdiYoutube} size={1.6} />
            </div>
            <h3>यह स्पॉटलाइट रिक्त है — इसका पहला सितारा बनें.</h3>
            <p>
              एक Rayforge ट्यूटोरियल बनाएँ, और हम इसे आपके नाम और चैनल लिंक के
              साथ सीधे मुख्य पृष्ठ पर प्रदर्शित करेंगे.
            </p>
            <Link to="/contributing" className={styles.buttonDark}>
              <Icon path={mdiPlayCircleOutline} size={0.85} />
              <span>पहला ट्यूटोरियल बनाएँ</span>
            </Link>
          </div>
        )}
      </div>
    </section>
  );
}

function CommunitySection() {
  return (
    <section className={styles.communitySection}>
      <div className={styles.communityInner}>
        <p className={styles.kicker}>शोकेस</p>
        <h2 className={styles.communityTitle}>Rayforge के साथ निर्मित</h2>
        <p className={styles.communitySubtitle}>
          देखें कि दुनिया भर के निर्माता क्या बना रहे हैं और अपना काम साझा करें.
        </p>
        <a
          href="https://discord.gg/sTHNdTtpQJ"
          className={styles.buttonDark}
          target="_blank"
          rel="noopener noreferrer"
        >
          <Icon path={mdiShareVariant} size={0.85} />
          <span>अपनी रचनाएँ साझा करें</span>
        </a>
      </div>
    </section>
  );
}

function ReferencesSection() {
  return (
    <section className={styles.refsSection}>
      <div className={styles.refsInner}>
        <div className={styles.refsHeader}>
          <p className={styles.kicker}>संदर्भ</p>
          <h2 className={styles.refsTitle}>वास्तविक कार्यशालाओं का विश्वास</h2>
          <p className={styles.refsSubtitle}>
            वे व्यवसाय जो अपने दैनिक काम के लिए Rayforge पर निर्भर करते हैं.
          </p>
        </div>

        <div className={styles.refsGrid}>
          {references.map((ref) => (
            <a
              key={ref.id}
              href={ref.url}
              target="_blank"
              rel="noopener noreferrer"
              className={styles.refCard}
            >
              <div className={styles.refLogo}>
                <img src={ref.logo} alt={`${ref.name} logo`} loading="lazy" />
              </div>
              <h3 className={styles.refName}>{ref.name}</h3>
              <span className={styles.refMeta}>{ref.tagline}</span>
              <p className={styles.refQuote}>&ldquo;{ref.quote}&rdquo;</p>
              <span className={styles.refLink}>
                <span>{ref.linkLabel}</span>
                <Icon path={mdiArrowRight} size={0.65} />
              </span>
            </a>
          ))}
        </div>
      </div>
    </section>
  );
}

export default function Home() {
  return (
    <Layout
      title="Free Open Source Laser Cutter Software"
      description="Rayforge is free open-source laser cutter and engraving software for GRBL-based machines. Design with AI, simulate in 3D, and control your laser — the LightBurn alternative."
    >
      <main className={styles.pageWrapper}>

        <HeroSection />

        <CapabilityStrip />

        <DesignYourPartsSection />

        <FeatureCardsSection />

        <TutorialSpotlight />

        <CommunitySection />

        <ReferencesSection />

      </main>
    </Layout>
  );
}
