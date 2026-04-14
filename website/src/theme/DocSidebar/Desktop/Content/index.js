import React from "react";
import SearchBar from "@theme-original/SearchBar";
import Content from "@theme-original/DocSidebar/Desktop/Content";
import styles from "./styles.module.css";

export default function DocSidebarDesktopContent(props) {
  return (
    <>
      <div className={styles.sidebarSearch}>
        <SearchBar />
      </div>
      <Content {...props} />
    </>
  );
}
