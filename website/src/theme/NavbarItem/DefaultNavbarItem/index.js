import React from 'react';
import DefaultNavbarItem from '@theme-original/NavbarItem/DefaultNavbarItem';
import {translate} from '@docusaurus/Translate';
import styles from './styles.module.css';

export default function NavbarItemWrapper(props) {
  if (props.type === 'localeDropdown' || props.type === 'search') {
    return <DefaultNavbarItem {...props} />;
  }
  if (props.label === 'GitHub') {
    if (props.mobile) {
      return (
        <li className="menu__list-item">
          <a
            href="https://www.patreon.com/c/knipknap"
            target="_blank"
            rel="noopener noreferrer"
            className="menu__link"
          >
            {translate({
              id: 'navbar.patreon.link',
              message: 'Become a Patron',
              description: 'Patreon link in mobile navbar',
            })}
          </a>
        </li>
      );
    }
    return (
      <>
        <a
          href="https://www.patreon.com/c/knipknap"
          target="_blank"
          rel="noopener noreferrer"
          className={styles.patreonLink}
        >
          <img
            src="https://c5.patreon.com/external/logo/become_a_patron_button.png"
            alt="Become a Patron"
            className={styles.patreonImg}
          />
        </a>
        <span className={styles.githubWrapper}>
          <DefaultNavbarItem {...props} />
        </span>
      </>
    );
  }
  if (props.label === 'Discord') {
    if (props.mobile) {
      return <DefaultNavbarItem {...props} />;
    }
    return (
      <span className={styles.discordWrapper}>
        <DefaultNavbarItem {...props} />
      </span>
    );
  }
  return <DefaultNavbarItem {...props} />;
}
