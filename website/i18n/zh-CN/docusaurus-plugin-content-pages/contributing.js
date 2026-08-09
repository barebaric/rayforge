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
  '第一次雕刻',
  '旋转轴雕刻设置',
  'AI 工件生成器',
  '相机校准',
  'Print & Cut 工作流程',
  '材料测试',
];

const quickActions = [
  {
    title: '报告错误',
    description: '打开一个 issue，包含重现步骤和您的预期结果。',
    href: 'https://github.com/barebaric/rayforge/issues/new',
    icon: mdiBugOutline,
    iconClass: styles.iconCyan,
  },
  {
    title: '建议功能',
    description: '分享您的使用场景以及成功的样子。',
    href: 'https://github.com/barebaric/rayforge/issues/new?labels=enhancement',
    icon: mdiLightbulbOnOutline,
    iconClass: styles.iconOrange,
  },
  {
    title: '提交代码',
    description: '按照开发者指南操作并发送 pull request。',
    to: '/docs/developer/getting-started',
    icon: mdiSourcePull,
    iconClass: styles.iconPurple,
  },
  {
    title: '改进文档',
    description: '修正错别字、添加示例，让文档更容易理解。',
    to: '/docs/getting-started/installation',
    icon: mdiBookOpenPageVariantOutline,
    iconClass: styles.iconCyan,
  },
  {
    title: '制作视频教程',
    description: '向世界传授 Rayforge——制作完成的教程会展示在首页。',
    href: '#video-tutorials',
    icon: mdiVideoOutline,
    iconClass: styles.iconRed,
    featured: true,
  },
];

export default function Contributing() {
  return (
    <Layout
      title="参与贡献"
      description="了解如何为 Rayforge 做出贡献：报告错误、建议功能、提交代码、制作视频教程、改进文档或为项目提供资金支持。"
    >
      <main className={styles.pageWrapper}>
        <section className={styles.hero}>
          <div className={styles.heroInner}>
            <div className={styles.heroContent}>
              <h1 className={styles.heroTitle}>
                为{' '}
                <span className={styles.heroTitleGradient}>Rayforge</span>{' '}
                做贡献
              </h1>
              <p className={styles.heroSubtitle}>
                帮助改进 Rayforge：报告错误、建议功能、提交代码、改进文档、
                制作教程或为项目提供资金支持。
              </p>
              <div className={styles.heroCtas}>
                <a
                  href="https://www.patreon.com/c/knipknap"
                  className={`rfButton rfButtonOrange ${styles.heroCtaButton}`}
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  <Icon path={mdiHandCoinOutline} size={0.9} />
                  <span>在 Patreon 上支持</span>
                </a>
                <a
                  href="https://github.com/barebaric/rayforge/issues/new"
                  className={`rfButton rfButtonDownload ${styles.heroCtaButton}`}
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  <Icon path={mdiBugOutline} size={0.9} />
                  <span>报告错误</span>
                </a>
                <Link
                  to="/docs/developer/getting-started"
                  className={`rfButton rfButtonPurple ${styles.heroCtaButton}`}
                >
                  <Icon path={mdiSourcePull} size={0.9} />
                  <span>开始贡献</span>
                </Link>
              </div>
            </div>

            <div className={styles.heroPanel}>
              <div className={styles.panelHeader}>
                <div className={styles.panelBadge}>
                  <Icon path={mdiGithub} size={0.85} />
                  <span>GitHub</span>
                </div>
                <h2 className={styles.panelTitle}>社区与支持</h2>
              </div>
              <div className={styles.panelLinks}>
                <a
                  href="https://github.com/barebaric/rayforge/issues"
                  className={styles.panelLink}
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  <span className={styles.panelLinkLabel}>报告问题</span>
                  <span className={styles.panelLinkMeta}>GitHub Issues</span>
                </a>
                <a
                  href="https://github.com/barebaric/rayforge"
                  className={styles.panelLink}
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  <span className={styles.panelLinkLabel}>浏览源代码</span>
                  <span className={styles.panelLinkMeta}>GitHub 仓库</span>
                </a>
                <Link to="/sponsor" className={styles.panelLink}>
                  <span className={styles.panelLinkLabel}>
                    成为赞助商
                  </span>
                  <span className={styles.panelLinkMeta}>帮助我们改进</span>
                </Link>
              </div>
            </div>
          </div>
        </section>

        <section className={styles.section}>
          <div className={styles.sectionInner}>
            <h2 className={styles.sectionTitle}>发挥最大作用</h2>
            <p className={styles.lead}>
              有些贡献比其他贡献更重要。目前，没有什么比视频教程更能帮助
              Rayforge 成长——而您的慷慨支持能让项目持续发展。
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
                  <span>最需要</span>
                </div>
                <div className={styles.impactCardHeader}>
                  <div className={`${styles.blockIcon} ${styles.iconRed}`}>
                    <Icon path={mdiYoutube} size={1.1} />
                  </div>
                  <h3 className={styles.impactCardTitle}>
                    制作视频教程
                  </h3>
                </div>
                <p className={styles.impactCardBody}>
                  视频是大多数人了解 Rayforge 的途径——制作完成的教程会展示在
                  首页，并附上您的名字和频道链接。
                </p>
                <ol className={styles.steps}>
                  <li className={styles.step}>
                    从下方选择一个主题——或者选择您自己的主题。
                  </li>
                  <li className={styles.step}>
                    录制一段带旁白的简短屏幕录像，上传到 YouTube，然后在{' '}
                    <a
                      href="https://discord.gg/sTHNdTtpQJ"
                      target="_blank"
                      rel="noopener noreferrer"
                    >
                      Discord
                    </a>{' '}
                    或{' '}
                    <a
                      href="https://github.com/barebaric/rayforge/discussions"
                      target="_blank"
                      rel="noopener noreferrer"
                    >
                      GitHub Discussions
                    </a>{' '}
                    上分享链接，即可获得展示位置。
                  </li>
                </ol>
                <div className={styles.wishlist}>
                  <div className={styles.wishlistTitle}>
                    <Icon path={mdiFire} size={0.85} />
                    <span>愿望清单——认领一个主题</span>
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
                  <span>分享您的教程</span>
                </a>
              </div>

              <div
                className={`${styles.impactCard} ${styles.impactSupport}`}
              >
                <div
                  className={`${styles.impactBadge} ${styles.impactBadgeSupport}`}
                >
                  <Icon path={mdiHandCoinOutline} size={0.8} />
                  <span>让项目保持活力</span>
                </div>
                <div className={styles.impactCardHeader}>
                  <div className={`${styles.blockIcon} ${styles.iconPurple}`}>
                    <Icon path={mdiHandCoinOutline} size={1.1} />
                  </div>
                  <h3 className={styles.impactCardTitle}>
                    资金支持
                  </h3>
                </div>
                <p className={styles.impactCardBody}>
                  Rayforge 是免费的，而且将永远免费。Patreon 和赞助资金用于支付
                  服务器、测试硬件和开发时间——它让项目不断前进。
                </p>
                <div className={styles.impactLinks}>
                  <a
                    href="https://www.patreon.com/c/knipknap"
                    className={`rfButton rfButtonOrange ${styles.impactCta}`}
                    target="_blank"
                    rel="noopener noreferrer"
                  >
                    <Icon path={mdiHandCoinOutline} size={0.9} />
                    <span>在 Patreon 上支持</span>
                  </a>
                  <Link
                    to="/sponsor"
                    className={`rfButton rfButtonPurple ${styles.impactCta}`}
                  >
                    <Icon path={mdiStarOutline} size={0.9} />
                    <span>成为赞助商</span>
                  </Link>
                </div>
              </div>
            </div>
          </div>
        </section>

        <section className={styles.section}>
          <div className={styles.sectionInner}>
            <h2 className={styles.sectionTitle}>快速操作</h2>
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
                        <span>获得展示</span>
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
            <h2 className={styles.sectionTitle}>贡献方式</h2>
            <p className={styles.lead}>
              我们欢迎各种形式的贡献。每一份错误报告、PR 和文档修正都会让
              Rayforge 对所有人更好。
            </p>

            <div className={styles.twoCol}>
              <div className={styles.block}>
                <div className={styles.blockHeader}>
                  <div className={`${styles.blockIcon} ${styles.iconCyan}`}>
                    <Icon path={mdiBugOutline} size={0.95} />
                  </div>
                  <h3 className={styles.blockTitle}>报告错误</h3>
                </div>
                <ol className={styles.steps}>
                  <li className={styles.step}>
                    查看{' '}
                    <a
                      href="https://github.com/barebaric/rayforge/issues"
                      target="_blank"
                      rel="noopener noreferrer"
                    >
                      现有问题
                    </a>{' '}
                    以避免重复。
                  </li>
                  <li className={styles.step}>
                    创建一个{' '}
                    <a
                      href="https://github.com/barebaric/rayforge/issues/new"
                      target="_blank"
                      rel="noopener noreferrer"
                    >
                      新问题
                    </a>{' '}
                    ，包含清晰的描述、重现步骤、预期行为与实际行为、系统信息，
                    以及适用时的截图。
                  </li>
                </ol>
              </div>

              <div className={styles.block}>
                <div className={styles.blockHeader}>
                  <div className={`${styles.blockIcon} ${styles.iconOrange}`}>
                    <Icon path={mdiLightbulbOnOutline} size={0.95} />
                  </div>
                  <h3 className={styles.blockTitle}>建议功能</h3>
                </div>
                <ol className={styles.steps}>
                  <li className={styles.step}>
                    查看{' '}
                    <a
                      href="https://github.com/barebaric/rayforge/issues?q=is%3Aissue+label%3Aenhancement"
                      target="_blank"
                      rel="noopener noreferrer"
                    >
                      现有功能请求
                    </a>
                    。
                  </li>
                  <li className={styles.step}>
                    打开一个功能请求，描述想法、使用场景、好处以及（可选）可能
                    的实现方式。
                  </li>
                </ol>
              </div>

              <div className={styles.block}>
                <div className={styles.blockHeader}>
                  <div className={`${styles.blockIcon} ${styles.iconPurple}`}>
                    <Icon path={mdiSourcePull} size={0.95} />
                  </div>
                  <h3 className={styles.blockTitle}>提交代码</h3>
                </div>
                <p className={styles.blockBody}>
                  有关提交代码贡献的详细信息，请参阅{' '}
                  <Link to="/docs/developer/getting-started">
                    开发者文档 – 快速入门
                  </Link>{' '}
                  指南。
                </p>
              </div>

              <div className={styles.block}>
                <div className={styles.blockHeader}>
                  <div className={`${styles.blockIcon} ${styles.iconCyan}`}>
                    <Icon path={mdiBookOpenPageVariantOutline} size={0.95} />
                  </div>
                  <h3 className={styles.blockTitle}>改进文档</h3>
                </div>
                <ul className={styles.bullets}>
                  <li>修正错别字或不清楚的解释</li>
                  <li>添加示例和截图</li>
                  <li>改进组织结构</li>
                  <li>翻译成其他语言</li>
                </ul>
                <p className={styles.blockBody}>
                  您可以点击任何文档页面上的“编辑此页”按钮，然后像提交代码
                  贡献一样提交 PR。
                </p>
              </div>
            </div>
          </div>
        </section>

        <section className={styles.section}>
          <div className={styles.sectionInner}>
            <h2 className={styles.sectionTitle}>关于本文档</h2>
            <p className={styles.lead}>
              本文档是为 Rayforge 的最终用户设计的。开发者文档请从这里开始：{' '}
              <Link to="/docs/developer/getting-started">开发者文档</Link>。
            </p>
          </div>
        </section>
      </main>
    </Layout>
  );
}
