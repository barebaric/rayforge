import React from 'react';
import useBrokenLinks from '@docusaurus/useBrokenLinks';

const ANCHORS = ['linux', 'windows', 'macos', 'linux-pixi', 'windows-developer'];

export default function InstallAnchors() {
  const brokenLinks = useBrokenLinks();
  return (
    <>
      {ANCHORS.map((id) => {
        brokenLinks.collectAnchor(id);
        return <span key={id} id={id} aria-hidden="true" />;
      })}
    </>
  );
}
