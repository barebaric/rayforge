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
  'Sua primeira gravação',
  'Configuração do eixo rotativo',
  'Gerador de peças com IA',
  'Calibração de câmera',
  'Fluxo de trabalho Print & Cut',
  'Testes de materiais',
];

const quickActions = [
  {
    title: 'Reportar um Bug',
    description: 'Abra um issue com passos para reproduzir e o que você esperava.',
    href: 'https://github.com/barebaric/rayforge/issues/new',
    icon: mdiBugOutline,
    iconClass: styles.iconCyan,
  },
  {
    title: 'Sugerir um Recurso',
    description: 'Compartilhe um caso de uso e como é o sucesso para você.',
    href: 'https://github.com/barebaric/rayforge/issues/new?labels=enhancement',
    icon: mdiLightbulbOnOutline,
    iconClass: styles.iconOrange,
  },
  {
    title: 'Enviar Código',
    description: 'Siga o guia do desenvolvedor e envie um pull request.',
    to: '/docs/developer/getting-started',
    icon: mdiSourcePull,
    iconClass: styles.iconPurple,
  },
  {
    title: 'Melhorar a Documentação',
    description: 'Corrija erros de digitação, adicione exemplos e torne a documentação mais fácil de seguir.',
    to: '/docs/getting-started/installation',
    icon: mdiBookOpenPageVariantOutline,
    iconClass: styles.iconCyan,
  },
  {
    title: 'Criar Tutoriais em Vídeo',
    description:
      'Ensine o Rayforge ao mundo — tutoriais concluídos são exibidos na página inicial.',
    href: '#video-tutorials',
    icon: mdiVideoOutline,
    iconClass: styles.iconRed,
    featured: true,
  },
];

export default function Contributing() {
  return (
    <Layout
      title="Contribuir"
      description="Aprenda como contribuir para o Rayforge: reporte bugs, sugira funcionalidades, envie código, crie tutoriais em vídeo, melhore a documentação ou apoie o projeto financeiramente."
    >
      <main className={styles.pageWrapper}>
        <section className={styles.hero}>
          <div className={styles.heroInner}>
            <div className={styles.heroContent}>
              <h1 className={styles.heroTitle}>
                Contribuindo para o{' '}
                <span className={styles.heroTitleGradient}>Rayforge</span>
              </h1>
              <p className={styles.heroSubtitle}>
                Ajude a melhorar o Rayforge: reporte bugs, sugira
                funcionalidades, envie código, refine a documentação, crie
                tutoriais ou apoie o projeto financeiramente.
              </p>
              <div className={styles.heroCtas}>
                <a
                  href="https://www.patreon.com/c/knipknap"
                  className={`rfButton rfButtonOrange ${styles.heroCtaButton}`}
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  <Icon path={mdiHandCoinOutline} size={0.9} />
                  <span>Apoiar no Patreon</span>
                </a>
                <a
                  href="https://github.com/barebaric/rayforge/issues/new"
                  className={`rfButton rfButtonDownload ${styles.heroCtaButton}`}
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  <Icon path={mdiBugOutline} size={0.9} />
                  <span>Reportar um Bug</span>
                </a>
                <Link
                  to="/docs/developer/getting-started"
                  className={`rfButton rfButtonPurple ${styles.heroCtaButton}`}
                >
                  <Icon path={mdiSourcePull} size={0.9} />
                  <span>Começar a Contribuir</span>
                </Link>
              </div>
            </div>

            <div className={styles.heroPanel}>
              <div className={styles.panelHeader}>
                <div className={styles.panelBadge}>
                  <Icon path={mdiGithub} size={0.85} />
                  <span>GitHub</span>
                </div>
                <h2 className={styles.panelTitle}>Comunidade e Suporte</h2>
              </div>
              <div className={styles.panelLinks}>
                <a
                  href="https://github.com/barebaric/rayforge/issues"
                  className={styles.panelLink}
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  <span className={styles.panelLinkLabel}>Reportar problemas</span>
                  <span className={styles.panelLinkMeta}>Issues no GitHub</span>
                </a>
                <a
                  href="https://github.com/barebaric/rayforge"
                  className={styles.panelLink}
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  <span className={styles.panelLinkLabel}>Explorar o código-fonte</span>
                  <span className={styles.panelLinkMeta}>Repositório GitHub</span>
                </a>
                <Link to="/sponsor" className={styles.panelLink}>
                  <span className={styles.panelLinkLabel}>
                    Torne-se um Patrocinador
                  </span>
                  <span className={styles.panelLinkMeta}>Ajude-nos a melhorar</span>
                </Link>
              </div>
            </div>
          </div>
        </section>

        <section className={styles.section}>
          <div className={styles.sectionInner}>
            <h2 className={styles.sectionTitle}>Cause o Maior Impacto</h2>
            <p className={styles.lead}>
              Algumas contribuições fazem mais diferença do que outras. Neste
              momento, nada ajuda o Rayforge a crescer tanto quanto tutoriais
              em vídeo — e sua generosidade mantém o projeto vivo.
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
                  <span>Muito Procurado</span>
                </div>
                <div className={styles.impactCardHeader}>
                  <div className={`${styles.blockIcon} ${styles.iconRed}`}>
                    <Icon path={mdiYoutube} size={1.1} />
                  </div>
                  <h3 className={styles.impactCardTitle}>
                    Crie Tutoriais em Vídeo
                  </h3>
                </div>
                <p className={styles.impactCardBody}>
                  Os vídeos são como a maioria das pessoas descobre o Rayforge
                  — e tutoriais concluídos são exibidos na página inicial com
                  seu nome e um link para o seu canal.
                </p>
                <ol className={styles.steps}>
                  <li className={styles.step}>
                    Escolha um tópico abaixo — ou crie o seu próprio.
                  </li>
                  <li className={styles.step}>
                    Grave uma captura de tela curta com narração, envie para o
                    YouTube e compartilhe o link no{' '}
                    <a
                      href="https://discord.gg/sTHNdTtpQJ"
                      target="_blank"
                      rel="noopener noreferrer"
                    >
                      Discord
                    </a>{' '}
                    ou nas{' '}
                    <a
                      href="https://github.com/barebaric/rayforge/discussions"
                      target="_blank"
                      rel="noopener noreferrer"
                    >
                      Discussões do GitHub
                    </a>{' '}
                    para garantir seu lugar.
                  </li>
                </ol>
                <div className={styles.wishlist}>
                  <div className={styles.wishlistTitle}>
                    <Icon path={mdiFire} size={0.85} />
                    <span>Lista de desejos — garanta um tópico</span>
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
                  <span>Compartilhe seu Tutorial</span>
                </a>
              </div>

              <div
                className={`${styles.impactCard} ${styles.impactSupport}`}
              >
                <div
                  className={`${styles.impactBadge} ${styles.impactBadgeSupport}`}
                >
                  <Icon path={mdiHandCoinOutline} size={0.8} />
                  <span>Mantém o Projeto Vivo</span>
                </div>
                <div className={styles.impactCardHeader}>
                  <div className={`${styles.blockIcon} ${styles.iconPurple}`}>
                    <Icon path={mdiHandCoinOutline} size={1.1} />
                  </div>
                  <h3 className={styles.impactCardTitle}>
                    Apoiar Financeiramente
                  </h3>
                </div>
                <p className={styles.impactCardBody}>
                  O Rayforge é gratuito e continuará sendo. O dinheiro do
                  Patreon e dos patrocínios paga servidores, hardware de teste
                  e tempo de desenvolvimento — ele mantém o projeto em
                  movimento.
                </p>
                <div className={styles.impactLinks}>
                  <a
                    href="https://www.patreon.com/c/knipknap"
                    className={`rfButton rfButtonOrange ${styles.impactCta}`}
                    target="_blank"
                    rel="noopener noreferrer"
                  >
                    <Icon path={mdiHandCoinOutline} size={0.9} />
                    <span>Apoiar no Patreon</span>
                  </a>
                  <Link
                    to="/sponsor"
                    className={`rfButton rfButtonPurple ${styles.impactCta}`}
                  >
                    <Icon path={mdiStarOutline} size={0.9} />
                    <span>Torne-se um Patrocinador</span>
                  </Link>
                </div>
              </div>
            </div>
          </div>
        </section>

        <section className={styles.section}>
          <div className={styles.sectionInner}>
            <h2 className={styles.sectionTitle}>Ações Rápidas</h2>
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
                        <span>Seja Destaque</span>
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
            <h2 className={styles.sectionTitle}>Formas de Contribuir</h2>
            <p className={styles.lead}>
              Aceitamos contribuições de todos os tipos. Cada relatório de bug,
              PR e correção de documentação torna o Rayforge melhor para todos.
            </p>

            <div className={styles.twoCol}>
              <div className={styles.block}>
                <div className={styles.blockHeader}>
                  <div className={`${styles.blockIcon} ${styles.iconCyan}`}>
                    <Icon path={mdiBugOutline} size={0.95} />
                  </div>
                  <h3 className={styles.blockTitle}>Reportar Bugs</h3>
                </div>
                <ol className={styles.steps}>
                  <li className={styles.step}>
                    Verifique os{' '}
                    <a
                      href="https://github.com/barebaric/rayforge/issues"
                      target="_blank"
                      rel="noopener noreferrer"
                    >
                      issues existentes
                    </a>{' '}
                    para evitar duplicatas.
                  </li>
                  <li className={styles.step}>
                    Crie um{' '}
                    <a
                      href="https://github.com/barebaric/rayforge/issues/new"
                      target="_blank"
                      rel="noopener noreferrer"
                    >
                      novo issue
                    </a>{' '}
                    com descrição clara, passos para reproduzir, comportamento
                    esperado vs. atual, informações do sistema e capturas de
                    tela, se aplicável.
                  </li>
                </ol>
              </div>

              <div className={styles.block}>
                <div className={styles.blockHeader}>
                  <div className={`${styles.blockIcon} ${styles.iconOrange}`}>
                    <Icon path={mdiLightbulbOnOutline} size={0.95} />
                  </div>
                  <h3 className={styles.blockTitle}>Sugerir Recursos</h3>
                </div>
                <ol className={styles.steps}>
                  <li className={styles.step}>
                    Verifique as{' '}
                    <a
                      href="https://github.com/barebaric/rayforge/issues?q=is%3Aissue+label%3Aenhancement"
                      target="_blank"
                      rel="noopener noreferrer"
                    >
                      solicitações de recursos existentes
                    </a>
                    .
                  </li>
                  <li className={styles.step}>
                    Abra uma solicitação de recurso descrevendo a ideia, o caso
                    de uso, os benefícios e (opcionalmente) uma abordagem
                    possível.
                  </li>
                </ol>
              </div>

              <div className={styles.block}>
                <div className={styles.blockHeader}>
                  <div className={`${styles.blockIcon} ${styles.iconPurple}`}>
                    <Icon path={mdiSourcePull} size={0.95} />
                  </div>
                  <h3 className={styles.blockTitle}>Enviar Código</h3>
                </div>
                <p className={styles.blockBody}>
                  Para informações detalhadas sobre o envio de contribuições de
                  código, consulte o guia{' '}
                  <Link to="/docs/developer/getting-started">
                    Documentação para Desenvolvedores – Primeiros Passos
                  </Link>{' '}
                  .
                </p>
              </div>

              <div className={styles.block}>
                <div className={styles.blockHeader}>
                  <div className={`${styles.blockIcon} ${styles.iconCyan}`}>
                    <Icon path={mdiBookOpenPageVariantOutline} size={0.95} />
                  </div>
                  <h3 className={styles.blockTitle}>Melhorar a Documentação</h3>
                </div>
                <ul className={styles.bullets}>
                  <li>Corrigir erros de digitação ou explicações confusas</li>
                  <li>Adicionar exemplos e capturas de tela</li>
                  <li>Melhorar a organização</li>
                  <li>Traduzir para outros idiomas</li>
                </ul>
                <p className={styles.blockBody}>
                  Você pode clicar no botão "editar esta página" em qualquer
                  página de documentação e depois enviar PRs da mesma forma que
                  contribuições de código.
                </p>
              </div>
            </div>
          </div>
        </section>

        <section className={styles.section}>
          <div className={styles.sectionInner}>
            <h2 className={styles.sectionTitle}>Sobre Esta Documentação</h2>
            <p className={styles.lead}>
              Esta documentação é projetada para usuários finais do Rayforge.
              Para a documentação de desenvolvedores, comece aqui:{' '}
              <Link to="/docs/developer/getting-started">Documentação para Desenvolvedores</Link>.
            </p>
          </div>
        </section>
      </main>
    </Layout>
  );
}
