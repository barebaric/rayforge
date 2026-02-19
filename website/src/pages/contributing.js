import React from 'react';
import Layout from '@theme/Layout';
import Link from '@docusaurus/Link';
import styles from './contributing.module.css';

export default function Contributing() {
  return (
    <Layout
      title="Contributing"
      description="Learn how to contribute to Rayforge"
    >
      <div className="container container--fluid margin-vert--lg">
        <div className="row">
          <div className="col col--8 col--offset-2">
            <h1>Contributing to Rayforge</h1>

            <div className={styles.supportSection}>
              <h2>Support the Project</h2>
              <a href="https://www.patreon.com/c/knipknap">
                <img
                  src="https://c5.patreon.com/external/logo/become_a_patron_button.png"
                  alt="Become a Patron"
                  height="55"
                />
              </a>
            </div>

            <h2>Community & Support</h2>

            <ul>
              <li>
                <strong>Report Issues</strong>:{' '}
                <a href="https://github.com/barebaric/rayforge/issues">
                  GitHub Issues
                </a>
              </li>
              <li>
                <strong>Source Code</strong>:{' '}
                <a href="https://github.com/barebaric/rayforge">
                  GitHub Repository
                </a>
              </li>
            </ul>

            <p>
              We welcome contributions of all kinds! Whether you're fixing
              bugs, adding features, improving documentation, or helping with
              packaging, your contributions make Rayforge better for everyone.
            </p>

            <h2>Ways to Contribute</h2>

            <h3>Report Bugs</h3>

            <p>Found a bug? Help us fix it:</p>

            <ol>
              <li>
                Check{' '}
                <a href="https://github.com/barebaric/rayforge/issues">
                  existing issues
                </a>{' '}
                to avoid duplicates
              </li>
              <li>
                Create a{' '}
                <a href="https://github.com/barebaric/rayforge/issues/new">
                  new issue
                </a>{' '}
                with:
                <ul>
                  <li>Clear description of the problem</li>
                  <li>Steps to reproduce</li>
                  <li>Expected vs. actual behavior</li>
                  <li>System information (OS, Rayforge version)</li>
                  <li>Screenshots or error messages if applicable</li>
                </ul>
              </li>
            </ol>

            <h3>Suggest Features</h3>

            <p>Have an idea for a new feature?</p>

            <ol>
              <li>
                Check{' '}
                <a href="https://github.com/barebaric/rayforge/issues?q=is%3Aissue+label%3Aenhancement">
                  existing feature requests
                </a>
              </li>
              <li>
                Open a feature request issue with:
                <ul>
                  <li>Description of the feature</li>
                  <li>Use case and benefits</li>
                  <li>Possible implementation approach (optional)</li>
                </ul>
              </li>
            </ol>

            <h3>Submit Code</h3>

            <p>
              For detailed information on submitting code contributions,
              please see the{' '}
              <Link to="/docs/developer/getting-started">
                Developer Documentation - Getting Started
              </Link>{' '}
              guide.
            </p>

            <h3>Improve Documentation</h3>

            <p>Documentation contributions are highly valued:</p>

            <ul>
              <li>Fix typos or unclear explanations</li>
              <li>Add examples and screenshots</li>
              <li>Improve organization</li>
              <li>Translate to other languages</li>
            </ul>

            <p>
              You can click the "edit this page" button on any documentation
              page, then submit PRs the same way as code contributions.
            </p>

            <h2>About This Documentation</h2>

            <p>
              This documentation is designed for end-users of Rayforge. If
              you're looking for developer documentation, please see the{' '}
              <Link to="/docs/developer/getting-started">
                Developer Documentation
              </Link>{' '}
              guide.
            </p>
          </div>
        </div>
      </div>
    </Layout>
  );
}
