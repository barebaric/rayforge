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
            Rayforge é um estúdio criativo completo para sua cortadora a laser,
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
    <Layout description="Projete e crie com sua cortadora a laser - o estúdio criativo completo para makers, artistas e artesãos">
      <main className={styles.pageWrapper}>
        
        <HeroSection />

        <QuickLinks />

        <div className={styles.featuresWrapper}>
          <FeatureSection
            blobClass={styles.blobPurple}
            title="Design com IA"
            description="Crie designs simplesmente descrevendo o que você quer. O Gerador de Peças por IA
              transforma suas descrições de texto em designs prontos para laser instantaneamente."
            features={[
              'Gere designs a partir de prompts de texto',
              'Não requer habilidades de design',
              'Funciona com qualquer provedor compatível com OpenAI',
            ]}
            image="/images/ai-prompt.png"
          />

          <FeatureSection
            blobClass={styles.blobCyan}
            title="Suas Criações em 3D"
            description="Transite perfeitamente de designs 2D para trajetórias 3D. O Rayforge gera G-code para cortadoras e gravadoras a laser de 2, 3 e 4 eixos, conectando designs planos a objetos físicos."
            compatibilityHeading="Compatível com: Grbl, Smoothieware"
            features={[
              'Simule seu trabalho em 3D',
              'Corte e grave objetos redondos',
              'Grave com passo descendente e eixos rotativos',
            ]}
            image="/images/screenshot-rotary-closeup.png"
            reverse
          />

          <FeatureSection
            blobClass={styles.blobOrange}
            title="Crie Suas Próprias Formas"
            description="Crie designs personalizados diretamente no Rayforge. As ferramentas de desenho integradas permitem esboçar, moldar e refinar suas ideias sem precisar de software de design separado."
            features={[
              'Desenhe linhas, círculos, curvas e formas preenchidas',
              'Alinhe tudo perfeitamente',
              'Defina dimensões que atualizam automaticamente',
            ]}
            image="/images/screenshot-sketcher.png"
          />

          <FeatureSection
            blobClass={styles.blobPurple}
            title="Caminhos de Corte Inteligentes"
            description="Trabalhe mais rápido e desperdice menos material. O Rayforge descobre a maneira mais eficiente de cortar seu design, para que seu laser passe menos tempo se movendo e mais tempo criando."
            features={[
              "Corte mais rápido com movimentos otimizados",
              'Economize material com layout automático',
              'Curvas suaves e precisas',
            ]}
            image="/images/screenshot-optimizer.png"
            reverse
          />

          <FeatureSection
            blobClass={styles.blobCyan}
            title="Visualize Antes de Cortar"
            description="Veja seu design ganhar vida virtualmente antes de se comprometer com o material real. Detecte problemas potenciais cedo e economize tempo e materiais."
            features={[
              'Veja exatamente como seu laser vai se mover',
              'Identifique erros antes de cortar',
              'Visualização em tempo real que atualiza enquanto você projeta',
            ]}
            image="/screenshots/main-simulation.png"
          />

          <FeatureSection
            blobClass={styles.blobOrange}
            title="Gerenciamento de Materiais e Fluxo de Trabalho"
            description="Salve suas configurações favoritas e obtenha resultados consistentes em todos os seus projetos. Seja cortando madeira, acrílico ou couro, o Rayforge lembra o que funciona melhor."
            features={[
              'Salve configurações para diferentes materiais',
              'Crie receitas reutilizáveis',
            ]}
            image="/images/screenshot-recipe.png"
            reverse
          />
        </div>

      </main>
    </Layout>
  );
}
