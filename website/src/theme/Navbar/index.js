import React from 'react';
import Navbar from '@theme-original/Navbar';
import styles from './styles.module.css';

function FloatingSocialLinks() {
  return (
    <div className={styles.floatingLinks}>
      <a
        className={styles.floatingLink}
        href="https://github.com/barebaric/rayforge"
        target="_blank"
        rel="noopener noreferrer"
        aria-label="Rayforge on GitHub"
        title="GitHub"
      >
        <img
          className={styles.floatingIcon}
          src="https://cdn.simpleicons.org/github/000000"
          alt=""
          width="20"
          height="20"
          loading="lazy"
        />
      </a>
      <a
        className={styles.floatingLink}
        href="https://discord.gg/sTHNdTtpQJ"
        target="_blank"
        rel="noopener noreferrer"
        aria-label="Join Rayforge on Discord"
        title="Discord"
      >
        <img
          className={styles.floatingIcon}
          src="https://cdn.simpleicons.org/discord/000000"
          alt=""
          width="20"
          height="20"
          loading="lazy"
        />
      </a>
      <a
        className={styles.floatingLink}
        href="https://www.patreon.com/c/knipknap"
        target="_blank"
        rel="noopener noreferrer"
        aria-label="Support Rayforge on Patreon"
        title="Patreon"
      >
        <img
          className={`${styles.floatingIcon} ${styles.floatingIconPatreon}`}
          src="https://cdn.simpleicons.org/patreon/FF424D"
          alt=""
          width="20"
          height="20"
          loading="lazy"
        />
      </a>
    </div>
  );
}

export default function NavbarWrapper(props) {
  return (
    <>
      <Navbar {...props} />
      <FloatingSocialLinks />
    </>
  );
}
