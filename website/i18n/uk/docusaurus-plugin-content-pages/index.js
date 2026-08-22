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
          <p className={styles.kicker}>Проєктування / Підготовка / Створення</p>
          <h1 className={styles.heroTitle}>
            <span className={styles.heroTitleLine1}>Від ідей</span>
            <span className={styles.heroTitleLine2}>до реальних проєктів</span>
          </h1>
          <p className={styles.heroSubtitle}>
            Rayforge — творчий набір для вашого лазерного верстата.
            Проєктуйте, готуйте та створюйте — усе в одній безкоштовній
            програмі з відкритим кодом.
          </p>
          <div className={styles.heroCtaButtons}>
            <Link to={downloadTo} className={styles.buttonDark}>
              <Icon path={mdiDownload} size={0.85} />
              <span>Завантажити безкоштовно</span>
            </Link>
            <a
              href="https://github.com/barebaric/rayforge"
              className={styles.buttonOutline}
              target="_blank"
              rel="noopener noreferrer"
            >
              <Icon path={mdiGithub} size={0.85} />
              <span>Відкритий код</span>
            </a>
          </div>
          <a
            href="https://www.youtube.com/watch?v=srKXs2p31VY"
            className={styles.heroVideoLink}
            target="_blank"
            rel="noopener noreferrer"
          >
            <Icon path={mdiPlayCircleOutline} size={0.9} />
            <span>Переглянути вступ</span>
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
    label: '2D CAD-скетчер',
    to: '/docs/features/sketcher',
  },
  {
    icon: mdiLayersOutline,
    label: 'Багатошарові завдання',
    to: '/docs/features/multi-layer',
  },
  {
    icon: mdiCameraOutline,
    label: 'Вирівнювання камерою',
    to: '/docs/machine/camera',
  },
  {
    icon: mdiRotate3d,
    label: 'Ротаційна вісь',
    to: '/docs/machine/rotary',
  },
  {
    icon: mdiBookOpenOutline,
    label: 'Рецепти матеріалів',
    to: '/docs/application-settings/recipes',
  },
  {
    icon: mdiMapOutline,
    label: 'Оптимізація шляхів',
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
            Потужні інструменти. Безмежні можливості.
          </p>
          <h2 className={styles.partsTitle}>Створюйте власні деталі</h2>
          <p className={styles.partsText}>
            Замальовуйте, формуйте та вдосконалюйте власні дизайни прямо в
            Rayforge. Вбудовані інструменти малювання дають життя будь-якій
            ідеї — або опишіть, що ви хочете, і AI-генератор заготовок створить
            це миттєво.
          </p>
          <Link to="/docs/features/sketcher" className={styles.partsLink}>
            <span>Дізнатися більше</span>
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
      title: 'Дизайн',
      subtitle: 'Потужний 2D CAD-скетчер з параметричними інструментами.',
      image: '/images/screenshot-sketcher.webp',
    },
    {
      title: 'Підготовка',
      subtitle:
        'Трасуйте зображення, оптимізуйте шляхи та тонко налаштовуйте кожну деталь.',
      image: '/images/screenshot-optimizer.webp',
    },
    {
      title: 'Створення',
      subtitle:
        'Запускайте лазерні та CNC-завдання впевнено. Швидко. Точно. Надійно.',
      image: '/screenshots/main-3d-bee.png',
    },
  ];

  return (
    <section className={styles.cardsSection}>
      <div className={styles.cardsTrees}>
        <div className={styles.cardsTreeLeft} />
        <div className={styles.cardsTreeRight} />
      </div>
      <div className={styles.cardsInner}>
        <p className={styles.cardsKicker}>
          Усе, що потрібно. Нічого зайвого.
        </p>
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
          <p className={styles.kicker}>Спільнота</p>
          <h2 className={styles.spotlightTitle}>
            Навчальні відео від справжніх користувачів
          </h2>
          <p className={styles.spotlightSubtitle}>
            Вчіться з відео, створених справжніми користувачами Rayforge. Ваше
            відео може з'явитися тут наступним.
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
            <h3>Ця вітрина порожня — станьте її першою зіркою.</h3>
            <p>
              Створіть навчальне відео про Rayforge, і ми розмістимо його прямо
              тут, на головній сторінці, із вашим ім'ям та посиланням на ваш
              канал.
            </p>
            <Link to="/contributing" className={styles.buttonDark}>
              <Icon path={mdiPlayCircleOutline} size={0.85} />
              <span>Створіть перше відео</span>
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
        <p className={styles.kicker}>Вітрина</p>
        <h2 className={styles.communityTitle}>Створено з Rayforge</h2>
        <p className={styles.communitySubtitle}>
          Дивіться, що створють автори по всьому світу, і діліться власними
          роботами.
        </p>
        <a
          href="https://discord.gg/sTHNdTtpQJ"
          className={styles.buttonDark}
          target="_blank"
          rel="noopener noreferrer"
        >
          <Icon path={mdiShareVariant} size={0.85} />
          <span>Поділіться своїми роботами</span>
        </a>
      </div>
    </section>
  );
}

export default function Home() {
  return (
    <Layout
      title="Безкоштовне програмне забезпечення з відкритим кодом для лазерного різання"
      description="Rayforge — безкоштовне програмне забезпечення з відкритим кодом для лазерного різання та гравіювання на верстатах з GRBL. Проєктуйте з AI, симулюйте у 3D та керуйте лазером — альтернатива LightBurn."
    >
      <main className={styles.pageWrapper}>

        <HeroSection />

        <CapabilityStrip />

        <DesignYourPartsSection />

        <FeatureCardsSection />

        <TutorialSpotlight />

        <CommunitySection />

      </main>
    </Layout>
  );
}
