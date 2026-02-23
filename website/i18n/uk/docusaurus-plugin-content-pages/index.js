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
            Безшовні інструменти для творчих людей. Rayforge — це потужний інструмент з відкритим кодом 
            для вашого лазерного різака, який об'єднує 2D CAD, CAM та керування машиною в одному динамічному пакеті.
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
    <Layout description="Сучасне програмне забезпечення для відправки G-коду та керування GRBL-сумісними лазерними різаками">
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
            title="Параметричний 2D-ескізник"
            description="Створюйте точні, засновані на обмеженнях 2D-дизайни безпосередньо в Rayforge. Параметричний ескізник пропонує повний набір інструментів для створення геометричних форм та застосування параметричних обмежень."
            features={[
              'Створення ліній, кіл, дуг та заливок',
              'Застосування геометричних обмежень: співпадіння, вертикаль, горизонталь...',
              'Параметричні вирази для обчислюваних розмірів',
            ]}
            image="/assets/screenshot-sketcher.png"
            reverse
          />

          <FeatureSection
            blobClass={styles.blobPurple}
            title="Розширена оптимізація траєкторій"
            description="Економте час та матеріал за допомогою інтелектуальних алгоритмів пошуку шляху та автоматичного компонування. Rayforge автоматично оптимізує порядок різання для мінімізації часу переміщення."
            features={[
              'Не витрачайте час на різання та гравіювання',
              'Економте матеріал завдяки автоматичному компонуванню та вкладенню',
              'Підтримка дуг з командами G2/G3',
            ]}
            image="/assets/screenshot-optimizer.png"
          />

          <FeatureSection
            blobClass={styles.blobCyan}
            title="Потужний вбудований симулятор"
            description="Перевірте свій G-код та симулюйте процес різання перед початком роботи. Наш вбудований симулятор допомагає виявити потенційні проблеми на ранній стадії, економлячи час та матеріал."
            features={[
              'Попередній перегляд G-коду',
              'Візуальна симуляція траєкторій різання',
              'Інтерактивний 2D-попередній перегляд усіх операцій під час обробки',
            ]}
            image="/screenshots/main-simulation.png"
            reverse
          />

          <FeatureSection
            blobClass={styles.blobOrange}
            title="Керування матеріалами та робочими процесами"
            description="Оптимізуйте свій робочий процес лазерного різання за допомогою комплексних інструментів для керування матеріалами, заготовками та рецептами, розробленими для підвищення узгодженості та ефективності."
            features={[
              'Організація матеріалів за типом та властивостями',
              'Визначення фізичних розмірів заготовок',
              'Створення повторно використовуваних рецептів для стабільних результатів',
            ]}
            image="/assets/screenshot-recipe.png"
          />
        </div>

      </main>
    </Layout>
  );
}
