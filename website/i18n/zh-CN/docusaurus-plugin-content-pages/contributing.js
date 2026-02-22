import React from 'react';
import Layout from '@theme/Layout';
import Link from '@docusaurus/Link';
import styles from '@site/src/pages/contributing.module.css';

export default function Contributing() {
  return (
    <Layout
      title="参与贡献"
      description="了解如何为 Rayforge 做出贡献"
    >
      <div className="container container--fluid margin-vert--lg">
        <div className="row">
          <div className="col col--8 col--offset-2">
            <h1>为 Rayforge 做贡献</h1>

            <div className={styles.supportSection}>
              <h2>支持本项目</h2>
              <a href="https://www.patreon.com/c/knipknap">
                <img
                  src="https://c5.patreon.com/external/logo/become_a_patron_button.png"
                  alt="成为赞助人"
                  height="55"
                />
              </a>
            </div>

            <h2>社区与支持</h2>

            <ul>
              <li>
                <strong>报告问题</strong>：{' '}
                <a href="https://github.com/barebaric/rayforge/issues">
                  GitHub Issues
                </a>
              </li>
              <li>
                <strong>源代码</strong>：{' '}
                <a href="https://github.com/barebaric/rayforge">
                  GitHub 仓库
                </a>
              </li>
            </ul>

            <p>
              我们欢迎各种形式的贡献！无论您是修复错误、添加功能、改进文档，还是帮助打包，您的贡献都会让 Rayforge 变得更好。
            </p>

            <h2>贡献方式</h2>

            <h3>报告错误</h3>

            <p>发现了错误？帮我们修复它：</p>

            <ol>
              <li>
                查看{' '}
                <a href="https://github.com/barebaric/rayforge/issues">
                  现有问题
                </a>{' '}
                以避免重复
              </li>
              <li>
                创建一个{' '}
                <a href="https://github.com/barebaric/rayforge/issues/new">
                  新问题
                </a>{' '}
                并包含：
                <ul>
                  <li>问题的清晰描述</li>
                  <li>重现步骤</li>
                  <li>预期行为与实际行为</li>
                  <li>系统信息（操作系统、Rayforge 版本）</li>
                  <li>如适用，请提供截图或错误信息</li>
                </ul>
              </li>
            </ol>

            <h3>建议功能</h3>

            <p>有新功能的想法？</p>

            <ol>
              <li>
                查看{' '}
                <a href="https://github.com/barebaric/rayforge/issues?q=is%3Aissue+label%3Aenhancement">
                  现有功能请求
                </a>
              </li>
              <li>
                提交功能请求问题并包含：
                <ul>
                  <li>功能描述</li>
                  <li>使用场景和好处</li>
                  <li>可能的实现方式（可选）</li>
                </ul>
              </li>
            </ol>

            <h3>提交代码</h3>

            <p>
              有关提交代码贡献的详细信息，请参阅{' '}
              <Link to="/docs/developer/getting-started">
                开发者文档 - 快速入门
              </Link>{' '}
              指南。
            </p>

            <h3>改进文档</h3>

            <p>我们非常重视文档贡献：</p>

            <ul>
              <li>修正错别字或不清楚的解释</li>
              <li>添加示例和截图</li>
              <li>改进组织结构</li>
              <li>翻译成其他语言</li>
            </ul>

            <p>
              您可以点击任何文档页面上的"编辑此页"按钮，然后像提交代码贡献一样提交 PR。
            </p>

            <h2>关于本文档</h2>

            <p>
              本文档是为 Rayforge 的最终用户设计的。如果您正在寻找开发者文档，请参阅{' '}
              <Link to="/docs/developer/getting-started">
                开发者文档
              </Link>{' '}
              指南。
            </p>
          </div>
        </div>
      </div>
    </Layout>
  );
}
