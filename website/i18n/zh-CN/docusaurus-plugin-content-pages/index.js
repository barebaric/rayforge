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
            为富有想象力的头脑提供无缝工具。Rayforge 是您的激光切割机的开源强力工具，将 2D CAD、CAM 和机器控制集成在一个充满活力的软件包中。
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
    <Layout description="基于 GRBL 的激光切割机现代 G-code 发送器和控制软件">
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
            title="参数化 2D 绘图器"
            description="直接在 Rayforge 中创建精确的、基于约束的 2D 设计。参数化绘图器提供了一套完整的工具，用于创建几何形状并应用参数化约束。"
            features={[
              '创建直线、圆形、圆弧和填充',
              '应用几何约束：重合、垂直、水平...',
              '用于计算尺寸的参数化表达式',
            ]}
            image="/assets/screenshot-sketcher.png"
            reverse
          />

          <FeatureSection
            blobClass={styles.blobPurple}
            title="高级路径优化"
            description="通过智能寻路算法和自动布局功能节省时间和材料。Rayforge 自动优化切割顺序，以最大限度地减少空走时间。"
            features={[
              "切割和雕刻时不浪费时间",
              '通过自动布局和排版节省材料',
              '原生支持 G2/G3 圆弧指令',
            ]}
            image="/assets/screenshot-optimizer.png"
          />

          <FeatureSection
            blobClass={styles.blobCyan}
            title="强大的内置模拟器"
            description="在开始之前验证您的 G-code 并模拟切割过程。我们的内置模拟器可帮助您及早发现潜在问题，节省您的时间和材料。"
            features={[
              'G-code 预览',
              '切割路径可视化模拟',
              '编辑时实时 2D 预览所有操作',
            ]}
            image="/screenshots/main-simulation.png"
            reverse
          />

          <FeatureSection
            blobClass={styles.blobOrange}
            title="材料和工作流管理"
            description="通过全面的材料、库存和配方管理工具简化您的激光切割工作流程，旨在提高一致性和效率。"
            features={[
              '按类型和属性组织材料',
              '定义物理库存尺寸',
              '创建可重复使用的配方以获得一致的结果',
            ]}
            image="/assets/screenshot-recipe.png"
          />
        </div>

      </main>
    </Layout>
  );
}
