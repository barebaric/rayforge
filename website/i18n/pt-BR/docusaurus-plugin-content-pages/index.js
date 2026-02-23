import React from 'react';
import Layout from '@theme/Layout';
import Link from '@docusaurus/Link';
import useBaseUrl from '@docusaurus/useBaseUrl';
import styles from '@site/src/pages/index.module.css';
import Icon from '@mdi/react';
import { mdiRocketLaunch, mdiViewDashboard, mdiFeatureSearch } from '@mdi/js';

function HeroSection() {
  return (
    <section className={styles.hero}>
      <div className={styles.heroInner}>
        
        <div className={styles.heroContent}>
          <h1 className={styles.heroTitle}>
            O <span className={styles.titleGradient}>Software Laser de Código Aberto Mais Poderoso</span> do Mundo
          </h1>
          <p className={styles.heroSubtitle}>
            Ferramentas perfeitas para mentes criativas. Rayforge é a ferramenta de código aberto para sua cortadora a laser,
            combinando CAD 2D, CAM e controle de máquina em um único pacote vibrante.
          </p>
          <div className={styles.heroCtaButtons}>
            <Link
              to="/docs/getting-started/installation"
              className={styles.buttonPrimary}
            >
              Começar
            </Link>
            <a
              href="https://github.com/barebaric/rayforge"
              className={styles.buttonSecondary}
              target="_blank"
              rel="noopener noreferrer"
            >
              Ver no GitHub
            </a>
          </div>
        </div>

        <div className={styles.heroVisuals}>
          <div className={styles.visualBlob}></div>
          
          <div className={styles.screenshotLayer}>
            <img
              src={useBaseUrl('/screenshots/main-standard.png')}
              alt="Captura de tela do aplicativo Rayforge"
              className={styles.heroImage}
            />
          </div>

          <div className={styles.videoLayer}>
            <div className={styles.videoWrapper}>
              <iframe
                src="https://www.youtube.com/embed/srKXs2p31VY"
                title="Introdução ao Rayforge"
                frameBorder="0"
                allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                allowFullScreen>
              </iframe>
            </div>
          </div>
        </div>

      </div>
    </section>
  );
}

function QuickLinks() {
  return (
    <div className={styles.linksContainer}>
      <div className={styles.cardGrid}>
        <Link to="/docs/getting-started/installation" className={styles.cardLink}>
          <div className={`${styles.card} ${styles.cardCyan}`}>
            <div className={styles.cardIcon}>
              <Icon path={mdiRocketLaunch} size={1.5} />
            </div>
            <h3>Primeiros Passos</h3>
            <p>Instale e configure o Rayforge.</p>
          </div>
        </Link>
        
        <Link to="/docs/ui/main-window" className={styles.cardLink}>
          <div className={`${styles.card} ${styles.cardPurple}`}>
            <div className={styles.cardIcon}>
              <Icon path={mdiViewDashboard} size={1.5} />
            </div>
            <h3>Interface do Usuário</h3>
            <p>Explore as ferramentas e o layout.</p>
          </div>
        </Link>
        
        <Link to="/docs/features/sketcher" className={styles.cardLink}>
          <div className={`${styles.card} ${styles.cardOrange}`}>
            <div className={styles.cardIcon}>
              <Icon path={mdiFeatureSearch} size={1.5} />
            </div>
            <h3>Recursos</h3>
            <p>Descubra mais funcionalidades.</p>
          </div>
        </Link>
      </div>
    </div>
  );
}

function FeatureSection({ 
  title, 
  description, 
  features, 
  image, 
  reverse, 
  compatibilityHeading,
  blobClass 
}) {
  return (
    <section className={styles.featureSection}>
      <div className={`${styles.featureSectionInner} ${reverse ? styles.reverse : ''}`}>
        
        <div className={styles.featureVisual}>
          <div className={`${styles.blobShape} ${blobClass}`}></div>
          <img src={useBaseUrl(image)} alt={title} />
        </div>

        <div className={styles.featureContent}>
          <h2>{title}</h2>
          <p>{description}</p>
          {compatibilityHeading && <p className={styles.compatibilityHeading}>{compatibilityHeading}</p>}
          {features && (
            <ul>
              {features.map((f, i) => <li key={i}>{f}</li>)}
            </ul>
          )}
        </div>
        
      </div>
    </section>
  );
}

export default function Home() {
  return (
    <Layout description="Software moderno de envio de G-code e controle para cortadoras e gravadoras a laser baseadas em GRBL">
      <main className={styles.pageWrapper}>
        
        <HeroSection />

        <QuickLinks />

        <div className={styles.featuresWrapper}>
          <FeatureSection
            blobClass={styles.blobCyan}
            title="Suas Criações em 3D"
            description="Transite perfeitamente de designs 2D para trajetórias 3D. O Rayforge gera G-code para cortadoras e gravadoras a laser de 2 e 3 eixos, conectando designs planos a objetos físicos."
            compatibilityHeading="Compatível com: Grbl, Smoothieware"
            features={['Visualização 3D', 'Geração de Trajetória', 'Controle de Eixos']}
            image="/assets/screenshot-3d-closeup.png"
          />

          <FeatureSection
            blobClass={styles.blobOrange}
            title="Desenhista 2D Paramétrico"
            description="Crie designs 2D precisos baseados em restrições diretamente no Rayforge. O desenhista paramétrico fornece um conjunto completo de ferramentas para criar formas geométricas e aplicar restrições paramétricas."
            features={[
              'Criar linhas, círculos, arcos e preenchimentos',
              'Aplicar restrições geométricas: coincidente, vertical, horizontal...',
              'Expressões paramétricas para dimensões calculadas',
            ]}
            image="/assets/screenshot-sketcher.png"
            reverse
          />

          <FeatureSection
            blobClass={styles.blobPurple}
            title="Otimização Avançada de Caminhos"
            description="Economize tempo e material com algoritmos inteligentes de planejamento de caminhos e funcionalidade de layout automático. O Rayforge otimiza automaticamente a ordem de corte para minimizar o tempo de deslocamento."
            features={[
              "Não perca tempo ao cortar e gravar",
              'Economize material com layout e encaixe automáticos',
              'Suporte nativo a arcos com comandos G2/G3',
            ]}
            image="/assets/screenshot-optimizer.png"
          />

          <FeatureSection
            blobClass={styles.blobCyan}
            title="Simulador Integrado Poderoso"
            description="Valide seu G-code e simule o processo de corte antes de começar. Nosso simulador integrado ajuda a detectar problemas potenciais cedo, economizando tempo e materiais."
            features={[
              'Pré-visualização de G-code',
              'Simulação visual dos caminhos de corte',
              'Pré-visualização 2D em tempo real de todas as operações enquanto você edita',
            ]}
            image="/screenshots/main-simulation.png"
            reverse
          />

          <FeatureSection
            blobClass={styles.blobOrange}
            title="Gerenciamento de Materiais e Fluxo de Trabalho"
            description="Otimize seu fluxo de trabalho de corte a laser com ferramentas abrangentes de gerenciamento de materiais, estoque e receitas projetadas para melhorar consistência e eficiência."
            features={[
              'Organize materiais por tipo e propriedades',
              'Defina dimensões físicas do estoque',
              'Crie receitas reutilizáveis para resultados consistentes',
            ]}
            image="/assets/screenshot-recipe.png"
          />
        </div>

      </main>
    </Layout>
  );
}
