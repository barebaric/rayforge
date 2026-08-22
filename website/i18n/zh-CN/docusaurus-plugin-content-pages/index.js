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
          <p className={styles.kicker}>设计 / 准备 / 创作</p>
          <h1 className={styles.heroTitle}>
            <span className={styles.heroTitleLine1}>从创意出发</span>
            <span className={styles.heroTitleLine2}>做出真实作品</span>
          </h1>
          <p className={styles.heroSubtitle}>
            Rayforge 是你的激光切割机创作套件。设计、准备、制作——尽在这一款免费开源应用中。
          </p>
          <div className={styles.heroCtaButtons}>
            <Link to={downloadTo} className={styles.buttonDark}>
              <Icon path={mdiDownload} size={0.85} />
              <span>免费下载</span>
            </Link>
            <a
              href="https://github.com/barebaric/rayforge"
              className={styles.buttonOutline}
              target="_blank"
              rel="noopener noreferrer"
            >
              <Icon path={mdiGithub} size={0.85} />
              <span>开源</span>
            </a>
          </div>
          <a
            href="https://www.youtube.com/watch?v=srKXs2p31VY"
            className={styles.heroVideoLink}
            target="_blank"
            rel="noopener noreferrer"
          >
            <Icon path={mdiPlayCircleOutline} size={0.9} />
            <span>观看介绍视频</span>
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
    label: '2D CAD 绘图',
    to: '/docs/features/sketcher',
  },
  {
    icon: mdiLayersOutline,
    label: '多层任务',
    to: '/docs/features/multi-layer',
  },
  {
    icon: mdiCameraOutline,
    label: '摄像头对位',
    to: '/docs/machine/camera',
  },
  {
    icon: mdiRotate3d,
    label: '旋转轴支持',
    to: '/docs/machine/rotary',
  },
  {
    icon: mdiBookOpenOutline,
    label: '材料配方',
    to: '/docs/application-settings/recipes',
  },
  {
    icon: mdiMapOutline,
    label: '路径优化',
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
          <p className={styles.partsKicker}>强大工具，无限可能。</p>
          <h2 className={styles.partsTitle}>制作你自己的零件</h2>
          <p className={styles.partsText}>
            在 Rayforge 中直接绘制、塑形并完善自定义设计。内置绘图工具让任何创意成为现实——也可以直接描述你想要的东西，AI 工件生成器会立即为你生成设计。
          </p>
          <Link to="/docs/features/sketcher" className={styles.partsLink}>
            <span>了解更多</span>
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
      title: '设计',
      subtitle: '强大的参数化 2D CAD 绘图工具。',
      image: '/images/screenshot-sketcher.png',
    },
    {
      title: '准备',
      subtitle: '描摹图像、优化加工路径，微调每一个细节。',
      image: '/images/screenshot-optimizer.png',
    },
    {
      title: '创作',
      subtitle: '自信地运行激光与 CNC 任务。快速。精准。可靠。',
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
        <p className={styles.cardsKicker}>应有尽有，绝不多余。</p>
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
          <p className={styles.kicker}>社区</p>
          <h2 className={styles.spotlightTitle}>真实用户制作的教程</h2>
          <p className={styles.spotlightSubtitle}>
            向 Rayforge 真实用户制作的视频学习。下一个出现在这里的可能就是你的教程。
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
            <h3>这里还空着——来做第一位主角吧。</h3>
            <p>
              制作一个 Rayforge 教程，我们会在首页展示它，附上你的名字和你的频道链接。
            </p>
            <Link to="/contributing" className={styles.buttonDark}>
              <Icon path={mdiPlayCircleOutline} size={0.85} />
              <span>制作第一个教程</span>
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
        <p className={styles.kicker}>作品展示</p>
        <h2 className={styles.communityTitle}>用 Rayforge 制作</h2>
        <p className={styles.communitySubtitle}>
          看看世界各地的创作者在做什么，并分享你自己的作品。
        </p>
        <a
          href="https://discord.gg/sTHNdTtpQJ"
          className={styles.buttonDark}
          target="_blank"
          rel="noopener noreferrer"
        >
          <Icon path={mdiShareVariant} size={0.85} />
          <span>分享你的作品</span>
        </a>
      </div>
    </section>
  );
}

export default function Home() {
  return (
    <Layout
      title="免费开源激光切割软件"
      description="Rayforge 是适用于 GRBL 机器的免费开源激光切割与雕刻软件。用 AI 设计、3D 模拟并控制你的激光机——LightBurn 的替代品。"
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
