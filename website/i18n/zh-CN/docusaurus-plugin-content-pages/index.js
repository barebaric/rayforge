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
        
        {/* Left Side: Original Content */}
        <div className={styles.heroContent}>
          <h1 className={styles.heroTitle}>
            全球<span className={styles.titleGradient}>最强大的开源</span>激光软件
          </h1>
          <p className={styles.heroSubtitle}>
            为富有想象力的头脑提供无缝工具。Rayforge 是您的激光切割机的完整创意工作室，将 2D CAD、CAM 和机器控制集成在一个充满活力的软件包中。
          </p>
          <div className={styles.heroCtaButtons}>
            <Link
              to="/docs/getting-started/installation"
              className={styles.buttonPrimary}
            >
              开始使用
            </Link>
            <a
              href="https://github.com/barebaric/rayforge"
              className={styles.buttonSecondary}
              target="_blank"
              rel="noopener noreferrer"
            >
              在 GitHub 上查看
            </a>
          </div>
        </div>

        {/* Right Side: Visual Cluster (Screenshot + Video) */}
        <div className={styles.heroVisuals}>
          <div className={styles.visualBlob}></div>
          
          {/* Main Back Layer: Screenshot */}
          <div className={styles.screenshotLayer}>
            <img
              src={useBaseUrl('/screenshots/main-standard.png')}
              alt="Rayforge 应用程序截图"
              className={styles.heroImage}
            />
          </div>

          {/* Floating Front Layer: Video */}
          <div className={styles.videoLayer}>
            <div className={styles.videoWrapper}>
              <iframe
                src="https://www.youtube.com/embed/srKXs2p31VY"
                title="Rayforge 介绍"
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
        {/* Cyan Theme Card */}
        <Link to="/docs/getting-started/installation" className={styles.cardLink}>
          <div className={`${styles.card} ${styles.cardCyan}`}>
            <div className={styles.cardIcon}>
              <Icon path={mdiRocketLaunch} size={1.5} />
            </div>
            <h3>快速入门</h3>
            <p>安装和配置 Rayforge。</p>
          </div>
        </Link>
        
        {/* Purple Theme Card */}
        <Link to="/docs/ui/main-window" className={styles.cardLink}>
          <div className={`${styles.card} ${styles.cardPurple}`}>
            <div className={styles.cardIcon}>
              <Icon path={mdiViewDashboard} size={1.5} />
            </div>
            <h3>用户界面</h3>
            <p>探索工具和布局。</p>
          </div>
        </Link>
        
        {/* Orange Theme Card */}
        <Link to="/docs/features/sketcher" className={styles.cardLink}>
          <div className={`${styles.card} ${styles.cardOrange}`}>
            <div className={styles.cardIcon}>
              <Icon path={mdiFeatureSearch} size={1.5} />
            </div>
            <h3>功能特性</h3>
            <p>发现更多功能。</p>
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
    <Layout description="使用激光切割机设计和创作 - 为创客、艺术家和手工艺人打造的完整创意工作室">
      <main className={styles.pageWrapper}>
        
        <HeroSection />

        <QuickLinks />

        <div className={styles.featuresWrapper}>
          <FeatureSection
            blobClass={styles.blobCyan}
            title="您的 3D 创作"
            description="无缝地从 2D 设计过渡到 3D 刀具路径。Rayforge 为 2 轴和 3 轴激光切割机和雕刻机生成 G-code，在平面设计和实体对象之间架起桥梁。"
            compatibilityHeading="兼容：Grbl、Smoothieware"
            features={['3D 可视化', '刀具路径生成', '轴控制']}
            image="/assets/screenshot-3d-closeup.png"
          />

          <FeatureSection
            blobClass={styles.blobOrange}
            title="设计您自己的形状"
            description="直接在 Rayforge 中创建自定义设计。内置绘图工具让您无需单独的设计软件即可绘制、塑造和完善您的创意。"
            features={[
              '绘制线条、圆形、曲线和填充形状',
              '完美对齐所有元素',
              '设置自动更新的尺寸',
            ]}
            image="/assets/screenshot-sketcher.png"
            reverse
          />

          <FeatureSection
            blobClass={styles.blobPurple}
            title="智能切割路径"
            description="更快地工作，更少地浪费材料。Rayforge 会找出切割您设计最高效的方式，让您的激光器花更少的时间移动，更多的时间创作。"
            features={[
              "通过优化的移动实现更快切割",
              '通过自动布局节省材料',
              '流畅、精确的曲线',
            ]}
            image="/assets/screenshot-optimizer.png"
          />

          <FeatureSection
            blobClass={styles.blobCyan}
            title="切割前预览"
            description="在实际切割之前，先虚拟地看到您的设计呈现效果。提前发现潜在问题，节省时间和材料。"
            features={[
              '准确查看激光的移动方式',
              '在切割前发现错误',
              '设计时实时预览更新',
            ]}
            image="/screenshots/main-simulation.png"
            reverse
          />

          <FeatureSection
            blobClass={styles.blobOrange}
            title="材料和工作流管理"
            description="保存您最喜爱的设置，在所有项目中获得一致的结果。无论您切割木材、亚克力还是皮革，Rayforge 都会记住最佳设置。"
            features={[
              '保存不同材料的设置',
              '创建可重复使用的配方',
            ]}
            image="/assets/screenshot-recipe.png"
          />
        </div>

      </main>
    </Layout>
  );
}
