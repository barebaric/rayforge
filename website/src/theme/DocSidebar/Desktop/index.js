import React from "react";
import clsx from "clsx";
import { useThemeConfig } from "@docusaurus/theme-common";
import Logo from "@theme/Logo";
import CollapseButton from "@theme/DocSidebar/Desktop/CollapseButton";
import SearchBar from "@theme-original/SearchBar";
import Content from "@theme-original/DocSidebar/Desktop/Content";
import styles from "@docusaurus/theme-classic/lib/theme/DocSidebar/Desktop/styles.module.css";
import sidebarStyles from "./styles.module.css";

export default function DocSidebarDesktop({
  path,
  sidebar,
  onCollapse,
  isHidden,
}) {
  const {
    navbar: { hideOnScroll },
    docs: {
      sidebar: { hideable },
    },
  } = useThemeConfig();
  return (
    <div
      className={clsx(
        styles.sidebar,
        hideOnScroll && styles.sidebarWithHideableNavbar,
        isHidden && styles.sidebarHidden,
      )}
    >
      {hideOnScroll && <Logo tabIndex={-1} className={styles.sidebarLogo} />}
      <div className={sidebarStyles.sidebarSearch}>
        <SearchBar />
      </div>
      <Content path={path} sidebar={sidebar} />
      {hideable && <CollapseButton onClick={onCollapse} />}
    </div>
  );
}
