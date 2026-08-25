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
  'Налаштування ротаційного модуля',
  'AI-генератор деталей',
  'Калібрування камери',
  'Робочий процес Print & Cut',
  'Тестування матеріалів',
];

const quickActions = [
  {
    title: 'Повідомити про помилку',
    description: 'Створіть issue з кроками відтворення та очікуваним результатом.',
    href: 'https://github.com/barebaric/rayforge/issues/new',
    icon: mdiBugOutline,
    iconClass: styles.iconCyan,
  },
  {
    title: 'Запропонувати функцію',
    description: 'Поділіться прикладом використання та тим, як виглядає успіх для вас.',
    href: 'https://github.com/barebaric/rayforge/issues/new?labels=enhancement',
    icon: mdiLightbulbOnOutline,
    iconClass: styles.iconOrange,
  },
  {
    title: 'Надіслати код',
    description: 'Дотримуйтесь посібника для розробників і надішліть pull request.',
    to: '/docs/developer/getting-started',
    icon: mdiSourcePull,
    iconClass: styles.iconPurple,
  },
  {
    title: 'Покращити документацію',
    description: 'Виправляйте помилки, додавайте приклади та робіть документацію зрозумілішою.',
    to: '/docs/getting-started/installation',
    icon: mdiBookOpenPageVariantOutline,
    iconClass: styles.iconCyan,
  },
];

export default function Contributing() {
  return (
    <Layout
      title="Участь у проекті"
      description="Дізнайтеся, як взяти участь у розробці Rayforge: повідомляйте про помилки, пропонуйте функції, надсилайте код, створюйте відеоуроки, покращуйте документацію або підтримуйте проєкт фінансово."
    >
      <main className={styles.pageWrapper}>
        <section className={styles.hero}>
          <div className={styles.heroInner}>
            <div className={styles.heroContent}>
              <h1 className={styles.heroTitle}>
                Участь у розробці{' '}
                <span className={styles.heroTitleGradient}>Rayforge</span>
              </h1>
              <p className={styles.heroSubtitle}>
                Допоможіть покращити Rayforge: повідомляйте про помилки,
                пропонуйте функції, надсилайте код, удосконалюйте
                документацію, створюйте відеоуроки або підтримуйте проєкт
                фінансово.
              </p>
              <div className={styles.heroCtas}>
                <a
                  href="https://www.patreon.com/c/knipknap"
                  className={`rfButton rfButtonOrange ${styles.heroCtaButton}`}
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  <Icon path={mdiHandCoinOutline} size={0.9} />
                  <span>Підтримати на Patreon</span>
                </a>
                <a
                  href="https://github.com/barebaric/rayforge/issues/new"
                  className={`rfButton rfButtonDownload ${styles.heroCtaButton}`}
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  <Icon path={mdiBugOutline} size={0.9} />
                  <span>Повідомити про помилку</span>
                </a>
                <Link
                  to="/docs/developer/getting-started"
                  className={`rfButton rfButtonPurple ${styles.heroCtaButton}`}
                >
                  <Icon path={mdiSourcePull} size={0.9} />
                  <span>Почати участь</span>
                </Link>
              </div>
            </div>

            <div className={styles.heroPanel}>
              <div className={styles.panelHeader}>
                <div className={styles.panelBadge}>
                  <Icon path={mdiGithub} size={0.85} />
                  <span>GitHub</span>
                </div>
                <h2 className={styles.panelTitle}>Спільнота та підтримка</h2>
              </div>
              <div className={styles.panelLinks}>
                <a
                  href="https://github.com/barebaric/rayforge/issues"
                  className={styles.panelLink}
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  <span className={styles.panelLinkLabel}>Повідомити про проблеми</span>
                  <span className={styles.panelLinkMeta}>GitHub Issues</span>
                </a>
                <a
                  href="https://github.com/barebaric/rayforge"
                  className={styles.panelLink}
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  <span className={styles.panelLinkLabel}>Переглянути вихідний код</span>
                  <span className={styles.panelLinkMeta}>Репозиторій GitHub</span>
                </a>
                <Link to="/sponsor" className={styles.panelLink}>
                  <span className={styles.panelLinkLabel}>
                    Стати спонсором
                  </span>
                  <span className={styles.panelLinkMeta}>Допоможіть нам стати кращими</span>
                </Link>
              </div>
            </div>
          </div>
        </section>

        <section className={styles.section}>
          <div className={styles.sectionInner}>
            <h2 className={styles.sectionTitle}>Зробіть найбільший внесок</h2>
            <p className={styles.lead}>
              Деякі внески впливають більше за інші. Зараз ніщо не допомагає
              Rayforge рости так, як відеоуроки, — а ваша щедрість підтримує
              проєкт.
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
                  <span>Найпотрібніше</span>
                </div>
                <div className={styles.impactCardHeader}>
                  <div className={`${styles.blockIcon} ${styles.iconRed}`}>
                    <Icon path={mdiYoutube} size={1.1} />
                  </div>
                  <h3 className={styles.impactCardTitle}>
                    Створюйте відеоуроки
                  </h3>
                </div>
                <p className={styles.impactCardBody}>
                  Відео — це те, як більшість людей відкривають для себе
                  Rayforge, — а готові уроки публікуються на головній
                  сторінці з вашим ім'ям і посиланням на канал.
                </p>
                <ol className={styles.steps}>
                  <li className={styles.step}>
                    Оберіть тему нижче — або придумайте свою.
                  </li>
                  <li className={styles.step}>
                    Запишіть короткий запис екрана з озвученням, завантажте
                    його на YouTube і поділіться посиланням на{' '}
                    <a
                      href="https://discord.gg/sTHNdTtpQJ"
                      target="_blank"
                      rel="noopener noreferrer"
                    >
                      Discord
                    </a>{' '}
                    або в{' '}
                    <a
                      href="https://github.com/barebaric/rayforge/discussions"
                      target="_blank"
                      rel="noopener noreferrer"
                    >
                      обговореннях GitHub
                    </a>{' '}
                    , щоб отримати своє місце.
                  </li>
                </ol>
                <div className={styles.wishlist}>
                  <div className={styles.wishlistTitle}>
                    <Icon path={mdiFire} size={0.85} />
                    <span>Список бажань — оберіть тему</span>
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
                  <span>Поділіться своїм уроком</span>
                </a>
              </div>

              <div
                className={`${styles.impactCard} ${styles.impactSupport}`}
              >
                <div
                  className={`${styles.impactBadge} ${styles.impactBadgeSupport}`}
                >
                  <Icon path={mdiHandCoinOutline} size={0.8} />
                  <span>Підтримує проєкт</span>
                </div>
                <div className={styles.impactCardHeader}>
                  <div className={`${styles.blockIcon} ${styles.iconPurple}`}>
                    <Icon path={mdiHandCoinOutline} size={1.1} />
                  </div>
                  <h3 className={styles.impactCardTitle}>
                    Підтримати фінансово
                  </h3>
                </div>
                <p className={styles.impactCardBody}>
                  Rayforge безкоштовний і залишиться таким. Кошти з Patreon і
                  спонсорство покривають сервери, тестове обладнання та час
                  розробки — це допомагає проєкту рухатися вперед.
                </p>
                <div className={styles.impactLinks}>
                  <a
                    href="https://www.patreon.com/c/knipknap"
                    className={`rfButton rfButtonOrange ${styles.impactCta}`}
                    target="_blank"
                    rel="noopener noreferrer"
                  >
                    <Icon path={mdiHandCoinOutline} size={0.9} />
                    <span>Підтримати на Patreon</span>
                  </a>
                  <Link
                    to="/sponsor"
                    className={`rfButton rfButtonPurple ${styles.impactCta}`}
                  >
                    <Icon path={mdiStarOutline} size={0.9} />
                    <span>Стати спонсором</span>
                  </Link>
                </div>
              </div>
            </div>
          </div>
        </section>

        <section className={styles.section}>
          <div className={styles.sectionInner}>
            <h2 className={styles.sectionTitle}>Швидкі дії</h2>
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
            <h2 className={styles.sectionTitle}>Про цю документацію</h2>
            <p className={styles.lead}>
              Ця документація призначена для кінцевих користувачів Rayforge.
              Для розробників почніть тут:{' '}
              <Link to="/docs/developer/getting-started">Документація для розробників</Link>.
            </p>
          </div>
        </section>
      </main>
    </Layout>
  );
}
