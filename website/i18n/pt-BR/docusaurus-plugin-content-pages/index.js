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
          <p className={styles.kicker}>Projetar / Preparar / Criar</p>
          <h1 className={styles.heroTitle}>
            <span className={styles.heroTitleLine1}>Das ideias</span>
            <span className={styles.heroTitleLine2}>a projetos reais</span>
          </h1>
          <p className={styles.heroSubtitle}>
            O Rayforge é o pacote criativo para a sua máquina de corte a laser.
            Projete, prepare e crie — tudo em um aplicativo gratuito e de
            código aberto.
          </p>
          <div className={styles.heroCtaButtons}>
            <Link to={downloadTo} className={styles.buttonDark}>
              <Icon path={mdiDownload} size={0.85} />
              <span>Baixe gratuitamente</span>
            </Link>
            <a
              href="https://github.com/barebaric/rayforge"
              className={styles.buttonOutline}
              target="_blank"
              rel="noopener noreferrer"
            >
              <Icon path={mdiGithub} size={0.85} />
              <span>Código aberto</span>
            </a>
          </div>
          <a
            href="https://www.youtube.com/watch?v=srKXs2p31VY"
            className={styles.heroVideoLink}
            target="_blank"
            rel="noopener noreferrer"
          >
            <Icon path={mdiPlayCircleOutline} size={0.9} />
            <span>Assista à introdução</span>
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
    label: 'Sketcher CAD 2D',
    to: '/docs/features/sketcher',
  },
  {
    icon: mdiLayersOutline,
    label: 'Trabalhos multicamadas',
    to: '/docs/features/multi-layer',
  },
  {
    icon: mdiCameraOutline,
    label: 'Alinhamento por câmera',
    to: '/docs/machine/camera',
  },
  {
    icon: mdiRotate3d,
    label: 'Eixo rotativo',
    to: '/docs/machine/rotary',
  },
  {
    icon: mdiBookOpenOutline,
    label: 'Receitas de materiais',
    to: '/docs/application-settings/recipes',
  },
  {
    icon: mdiMapOutline,
    label: 'Otimização de trajetos',
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
            Ferramentas poderosas. Possibilidades infinitas.
          </p>
          <h2 className={styles.partsTitle}>Crie suas próprias peças</h2>
          <p className={styles.partsText}>
            Desenhe, modele e refine projetos personalizados dentro do
            Rayforge. As ferramentas de desenho integradas dão vida a qualquer
            ideia — ou descreva o que você quer e o Gerador de Peças por IA
            cria tudo instantaneamente.
          </p>
          <Link to="/docs/features/sketcher" className={styles.partsLink}>
            <span>Saiba mais</span>
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
      title: 'Projeto',
      subtitle: 'Sketcher CAD 2D poderoso com ferramentas paramétricas.',
      image: '/images/screenshot-sketcher.webp',
    },
    {
      title: 'Preparar',
      subtitle:
        'Vetorize imagens, otimize trajetórias e ajuste cada detalhe.',
      image: '/images/screenshot-optimizer.webp',
    },
    {
      title: 'Criar',
      subtitle:
        'Execute trabalhos a laser e CNC com confiança. Rápido. Preciso. Confiável.',
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
        <p className={styles.cardsKicker}>
          Tudo o que você precisa. Nada supérfluo.
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
          <p className={styles.kicker}>Comunidade</p>
          <h2 className={styles.spotlightTitle}>
            Tutoriais de usuários reais
          </h2>
          <p className={styles.spotlightSubtitle}>
            Aprenda com vídeos feitos por usuários reais do Rayforge. Seu
            tutorial pode ser o próximo a aparecer aqui.
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
            <h3>Este destaque está vazio — seja a primeira estrela.</h3>
            <p>
              Faça um tutorial do Rayforge e nós o destacaremos aqui mesmo na
              página inicial, com seu nome e um link para o seu canal.
            </p>
            <Link to="/contributing" className={styles.buttonDark}>
              <Icon path={mdiPlayCircleOutline} size={0.85} />
              <span>Crie o primeiro tutorial</span>
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
        <p className={styles.kicker}>Vitrine</p>
        <h2 className={styles.communityTitle}>Feito com Rayforge</h2>
        <p className={styles.communitySubtitle}>
          Veja o que criadores do mundo inteiro estão fazendo e compartilhe o
          seu próprio trabalho.
        </p>
        <a
          href="https://discord.gg/sTHNdTtpQJ"
          className={styles.buttonDark}
          target="_blank"
          rel="noopener noreferrer"
        >
          <Icon path={mdiShareVariant} size={0.85} />
          <span>Compartilhe suas criações</span>
        </a>
      </div>
    </section>
  );
}

export default function Home() {
  return (
    <Layout
      title="Software livre e gratuito para corte a laser"
      description="Rayforge é um software gratuito e de código aberto de corte e gravação a laser para máquinas baseadas em GRBL. Projete com IA, simule em 3D e controle seu laser — a alternativa ao LightBurn."
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
