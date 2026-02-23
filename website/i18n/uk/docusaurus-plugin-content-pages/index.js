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
            <span className={styles.titleGradient}>Найпотужніше відкрите програмне забезпечення</span> для лазерів
          </h1>
          <p className={styles.heroSubtitle}>
            Безшовні інструменти для творчих людей. Rayforge — це повноцінна творча студія 
            для вашого лазерного різака, яка об'єднує 2D CAD, CAM та керування машиною в одному динамічному пакеті.
          </p>
          <div className={styles.heroCtaButtons}>
            <Link
              to="/docs/getting-started/installation"
              className={styles.buttonPrimary}
            >
              Почати
            </Link>
            <a
              href="https://github.com/barebaric/rayforge"
              className={styles.buttonSecondary}
              target="_blank"
              rel="noopener noreferrer"
            >
              Переглянути на GitHub
            </a>
          </div>
        </div>

        <div className={styles.heroVisuals}>
          <div className={styles.visualBlob}></div>
          
          <div className={styles.screenshotLayer}>
            <img
              src={useBaseUrl('/screenshots/main-standard.png')}
              alt="Знімок екрану програми Rayforge"
              className={styles.heroImage}
            />
          </div>

          <div className={styles.videoLayer}>
            <div className={styles.videoWrapper}>
              <iframe
                src="https://www.youtube.com/embed/srKXs2p31VY"
                title="Вступ до Rayforge"
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
            <h3>Початок роботи</h3>
            <p>Встановіть та налаштуйте Rayforge.</p>
          </div>
        </Link>
        
        <Link to="/docs/ui/main-window" className={styles.cardLink}>
          <div className={`${styles.card} ${styles.cardPurple}`}>
            <div className={styles.cardIcon}>
              <Icon path={mdiViewDashboard} size={1.5} />
            </div>
            <h3>Інтерфейс користувача</h3>
            <p>Ознайомтеся з інструментами та макетом.</p>
          </div>
        </Link>
        
        <Link to="/docs/features/sketcher" className={styles.cardLink}>
          <div className={`${styles.card} ${styles.cardOrange}`}>
            <div className={styles.cardIcon}>
              <Icon path={mdiFeatureSearch} size={1.5} />
            </div>
            <h3>Функції</h3>
            <p>Дослідіть більше можливостей.</p>
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
    <Layout description="Проектуйте та створюйте за допомогою лазерного різака — повноцінна творча студія для майстрів, художників та ремісників">
      <main className={styles.pageWrapper}>
        
        <HeroSection />

        <QuickLinks />

        <div className={styles.featuresWrapper}>
          <FeatureSection
            blobClass={styles.blobCyan}
            title="Ваші творіння в 3D"
            description="Безшовний перехід від 2D-дизайнів до 3D-траєкторій інструменту. Rayforge генерує G-код для 2-осьових та 3-осьових лазерних різаків і граверів, заповнюючи розрив між пласкими дизайнами та фізичними об'єктами."
            compatibilityHeading="Сумісно з: Grbl, Smoothieware"
            features={['3D-візуалізація', 'Генерація траєкторій інструменту', 'Керування осями']}
            image="/assets/screenshot-3d-closeup.png"
          />

          <FeatureSection
            blobClass={styles.blobOrange}
            title="Створюйте власні форми"
            description="Створюйте власні дизайни безпосередньо в Rayforge. Вбудовані інструменти малювання дозволяють створювати ескізи, форми та уточнювати ваші ідеї без необхідності в окремому програмному забезпеченні для дизайну."
            features={[
              'Малюйте лінії, кола, криві та заповнені форми',
              'Вирівнюйте все ідеально',
              'Встановлюйте розміри, що оновлюються автоматично',
            ]}
            image="/assets/screenshot-sketcher.png"
            reverse
          />

          <FeatureSection
            blobClass={styles.blobPurple}
            title="Розумні траєкторії різання"
            description="Працюйте швидше та витрачайте менше матеріалу. Rayforge визначає найефективніший спосіб різання вашого дизайну, щоб ваш лазер витрачав менше часу на переміщення та більше на творення."
            features={[
              'Швидше різання з оптимізованими переміщеннями',
              'Економте матеріал завдяки автоматичному компонуванню',
              'Плавні та точні криві',
            ]}
            image="/assets/screenshot-optimizer.png"
          />

          <FeatureSection
            blobClass={styles.blobCyan}
            title="Попередній перегляд перед різанням"
            description="Побачте, як ваш дизайн оживає віртуально, перш ніж різати справжній матеріал. Виявляйте потенційні проблеми на ранній стадії та економте час і матеріали."
            features={[
              'Бачте точно, як рухатиметься ваш лазер',
              'Виявляйте помилки перед різанням',
              'Інтерактивний попередній перегляд, що оновлюється під час дизайну',
            ]}
            image="/screenshots/main-simulation.png"
            reverse
          />

          <FeatureSection
            blobClass={styles.blobOrange}
            title="Керування матеріалами та робочими процесами"
            description="Зберігайте улюблені налаштування та отримуйте стабільні результати для всіх ваших проектів. Чи ріжете ви дерево, акрил чи шкіру, Rayforge пам'ятає, що працює найкраще."
            features={[
              'Зберігайте налаштування для різних матеріалів',
              'Створюйте рецепти для повторного використання',
            ]}
            image="/assets/screenshot-recipe.png"
          />
        </div>

      </main>
    </Layout>
  );
}
