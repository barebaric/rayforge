import React from 'react';
import Layout from '@theme/Layout';
import Link from '@docusaurus/Link';
import Translate, { translate } from '@docusaurus/Translate';
import Icon from '@mdi/react';
import {
  mdiUsbPort,
  mdiWifi,
  mdiChip,
  mdiCartOutline,
  mdiLaserPointer,
  mdiAlertOutline,
  mdiInformationOutline,
} from '@mdi/js';
import styles from './devices.module.css';

const SECTION_GROUP = {
  GrblSerialDriver: 'Grbl',
  GrblNetworkDriver: 'Grbl',
  SmoothieDriver: 'Smoothieware',
  MarlinSerialDriver: 'Marlin',
  RuidaDriver: 'Ruida',
  OctoPrintDriver: 'OctoPrint',
};

const SECTION_LABELS = {
  Grbl: { icon: mdiUsbPort, label: 'GRBL' },
  Smoothieware: { icon: mdiChip, label: 'Smoothieware' },
  Marlin: { icon: mdiChip, label: 'Marlin' },
  Ruida: { icon: mdiChip, label: 'Ruida' },
  OctoPrint: { icon: mdiWifi, label: 'OctoPrint' },
};

const SECTION_ORDER = ['Grbl', 'Smoothieware', 'Marlin', 'Ruida', 'OctoPrint'];

const GENERIC_NAMES = [
  'Generic GRBL',
  'Generic Ruida',
  'Generic Smoothieware',
];

const devices = [
  {
    id: 'acmer-s1',
    name: 'Acmer S1',
    description: 'Compact diode laser engraver with 130x130mm work area',
    driver: 'GrblSerialDriver',
    affiliate: {
      href: 'https://s.click.aliexpress.com/e/_c3oAKd75',
      img: 'https://ae01.alicdn.com/kf/Scaea4fcfd4e34286b75eaeb08f0479b6X.jpg_350x350.jpg',
      shop: 'AliExpress',
    },
  },
  {
    id: 'atomstack-a70',
    name: 'Atomstack A70',
    description: '70W diode laser cutter with 500x500mm work area and spot compression',
    driver: 'GrblSerialDriver',
    affiliate: {
      href: 'https://s.click.aliexpress.com/e/_c3fX3Orp',
      img: 'https://ae01.alicdn.com/kf/Se773bf3973fd40fc961c4b76b4c9c1b2J.jpg_350x350.jpg',
      shop: 'AliExpress',
    },
  },
  {
    id: 'atomstack-x40-pro',
    name: 'Atomstack X40 Pro',
    description: '40W diode laser engraver with 400x400mm work area and air assist',
    driver: 'GrblSerialDriver',
    affiliate: {
      href: 'https://s.click.aliexpress.com/e/_c2v1qAbh',
      img: 'https://ae-pic-a1.aliexpress-media.com/kf/S59613c6b301c4562a8471c2763f67f73p.jpg',
      shop: 'AliExpress',
    },
  },
  {
    id: 'carvera-air',
    name: 'Carvera Air',
    description: 'Desktop laser engraver with Smoothieware controller and 300x200mm work area',
    driver: 'SmoothieDriver',
  },
  {
    id: 'creality-falcon-2-pro',
    name: 'Creality Falcon 2 Pro 40W',
    description: 'Enclosed 40W diode laser engraver with 300x300mm work area',
    driver: 'GrblSerialDriver',
  },
  {
    id: 'creality-falcon-a1',
    name: 'Creality Falcon A1',
    description: 'Diode laser engraver with 381x305mm work area and rotary support',
    driver: 'GrblSerialDriver',
    affiliate: {
      href: 'https://s.click.aliexpress.com/e/_c4Cf2ILp',
      img: 'https://ae01.alicdn.com/kf/S440a905b44684a7a83900225c6b2789bx.png_350x350.png',
      shop: 'AliExpress',
    },
  },
  {
    id: 'elidor-z6',
    name: 'Elidor Z6',
    description: 'Diode laser engraver with 300x300mm work area',
    driver: 'GrblSerialDriver',
    affiliate: {
      href: 'https://s.click.aliexpress.com/e/_c3f1CH6j',
      img: 'https://ae01.alicdn.com/kf/S72dfe28563ef4ec08b1df4d01ba11b0b8.jpg_350x350.jpg',
      shop: 'AliExpress',
    },
  },
  {
    id: 'grbl-mks-dlc32',
    name: 'Grbl MKS DLC32',
    description: 'GRBL-based controller board for custom laser builds',
    driver: 'GrblSerialDriver',
    affiliate: {
      href: 'https://s.click.aliexpress.com/e/_c3jiLkgF',
      img: 'https://ae01.alicdn.com/kf/S91dcec5e84c04bb4969db3e651d52637A.jpg_350x350.jpg',
      shop: 'AliExpress',
    },
  },
  {
    id: 'longer-ray5',
    name: 'Longer Ray5 20W',
    description: 'Diode laser engraver with 375x375mm work area and touchscreen display',
    driver: 'GrblSerialDriver',
    affiliate: {
      href: 'https://s.click.aliexpress.com/e/_c4o81KkF',
      img: 'https://ae01.alicdn.com/kf/S5839c01e590349bbaae10394b5d9171aV.png_350x350.png',
      shop: 'AliExpress',
    },
  },
  {
    id: 'neje-master-3-max',
    name: 'NEJE Master 3 Max',
    description: '40W diode laser engraver with 460x460mm work area',
    driver: 'GrblSerialDriver',
    affiliate: {
      href: 'https://s.click.aliexpress.com/e/_c4XCDN1D',
      img: 'https://ae01.alicdn.com/kf/S86cbada3786e49488c2c54dabdc0f066U.jpg_350x350.jpg',
      shop: 'AliExpress',
    },
  },
  {
    id: 'octoprint',
    name: 'OctoPrint',
    description: 'A laser connected through an OctoPrint server',
    driver: 'OctoPrintDriver',
  },
  {
    id: 'omtech-k40plus',
    name: 'OMTech K40+',
    description: 'CO2 laser cutter with 300x200mm work area',
    driver: 'GrblSerialDriver',
  },
  {
    id: 'ortur-laser-master-3',
    name: 'Ortur Laser Master 3',
    description: 'Diode laser engraver with 400x400mm work area and 32-bit controller',
    driver: 'GrblSerialDriver',
    affiliate: {
      href: 'https://s.click.aliexpress.com/e/_c2Rx4f5p',
      img: 'https://ae01.alicdn.com/kf/S41720706142d4a86bf24cfc566456867p.jpg_350x350.jpg',
      shop: 'AliExpress',
    },
  },
  {
    id: 'ortur-laser-master-4',
    name: 'Ortur Laser Master 4',
    description: 'Diode laser engraver with 400x400mm work area and high-speed capabilities',
    driver: 'GrblSerialDriver',
  },
  {
    id: 'sculpfun-icube',
    name: 'Sculpfun iCube',
    description: 'Compact diode laser engraver with 120x120mm work area',
    driver: 'GrblSerialDriver',
    affiliate: {
      href: 'https://s.click.aliexpress.com/e/_c2JhZGGb',
      img: 'https://ae01.alicdn.com/kf/Sa68d14f02093443b82152e72b896bc3eT.jpg_350x350.jpg',
      shop: 'AliExpress',
    },
  },
  {
    id: 'sculpfun-icube-ultra',
    name: 'Sculpfun iCube Ultra',
    description: 'Diode laser engraver with Bluetooth and 150x150mm work area',
    driver: 'GrblSerialDriver',
    affiliate: {
      href: 'https://s.click.aliexpress.com/e/_c4BvYTnH',
      img: 'https://ae01.alicdn.com/kf/Sdd5cdb8a275d442899f6e9be16bd8921y.png_350x350.png',
      shop: 'AliExpress',
    },
  },
  {
    id: 'sculpfun-s30',
    name: 'Sculpfun S30',
    description: 'Budget diode laser engraver with 400x400mm work area',
    driver: 'GrblSerialDriver',
    affiliate: {
      href: 'https://s.click.aliexpress.com/e/_c3v7A7fV',
      img: 'https://ae01.alicdn.com/kf/S3fca59563a594c0582050a3481d7f986v.jpg_350x350.jpg',
      shop: 'AliExpress',
    },
  },
  {
    id: 'sculpfun-s30-pro-max',
    name: 'Sculpfun S30 Pro Max',
    description: '20W diode laser engraver with 370x360mm work area and automatic air-assist',
    driver: 'GrblSerialDriver',
    affiliate: {
      href: 'https://www.awin1.com/cread.php?awinmid=48841&awinaffid=2889815&ued=https%3A%2F%2Fwww.sculpfun.com%2Fcollections%2Fs30-series%2Fproducts%2Fsculpfun-s30-pro-max-laser-engraver-machine%3Fvariant%3D42446149845170',
      img: 'https://www.sculpfun.com/cdn/shop/files/S30-Pro-Max.jpg?v=1751342420&width=800',
      shop: 'Sculpfun',
    },
  },
  {
    id: 'sculpfun-s40-max',
    name: 'Sculpfun S40 MAX',
    description: '48W diode laser cutter with 830x800mm work area and auto-focus',
    driver: 'GrblSerialDriver',
    affiliate: {
      href: 'https://s.click.aliexpress.com/e/_c3OPMngB',
      img: 'https://ae01.alicdn.com/kf/S0a7d49df19a549f4be04c0e950ea0a31Q.jpg_350x350.jpg',
      shop: 'AliExpress',
    },
  },
  {
    id: 'sculpfun-s70-max',
    name: 'Sculpfun S70 MAX',
    description: '70W diode laser cutter with 830x800mm work area and auto-focus',
    driver: 'GrblSerialDriver',
    affiliate: {
      href: 'https://s.click.aliexpress.com/e/_c4e1ySiR',
      img: 'https://ae01.alicdn.com/kf/Scc68c784d03243c7ab5f3527e1d6fa9fH.jpg_350x350.jpg',
      shop: 'AliExpress',
    },
  },
  {
    id: 'twotrees-tts55',
    name: 'TwoTrees TTS-55',
    description: '55W diode laser engraver with 400x400mm work area',
    driver: 'GrblSerialDriver',
    affiliate: {
      href: 'https://s.click.aliexpress.com/e/_c3Pkh3cr',
      img: 'https://ae01.alicdn.com/kf/S21bb92cf45be4fd3834bf6dab3caeacaA.jpg_350x350.jpg',
      shop: 'AliExpress',
    },
  },
  {
    id: 'xtool-d1-pro',
    name: 'xTool D1 Pro',
    description: 'Diode laser engraver with 430x390mm work area, connected via Wi-Fi',
    driver: 'GrblNetworkDriver',
    affiliate: {
      href: 'https://s.click.aliexpress.com/e/_c4PqlR0n',
      img: 'https://ae01.alicdn.com/kf/S71096d813f5b4a329d25c12d1e1c6848U.jpg_350x350.jpg',
      shop: 'AliExpress',
    },
  },
];

const isGeneric = (d) => GENERIC_NAMES.includes(d.name);

const sections = SECTION_ORDER.map((groupName) => {
  const groupDevices = devices.filter(
    (d) => SECTION_GROUP[d.driver] === groupName,
  );
  return {
    groupName,
    label: SECTION_LABELS[groupName],
    cards: groupDevices.filter((d) => d.affiliate),
    list: groupDevices.filter((d) => !d.affiliate && !isGeneric(d)),
  };
});

const grblLink = (
  <a href="https://github.com/gnea/grbl" target="_blank" rel="noopener noreferrer">GRBL</a>
);

const smoothieLink = (
  <a href="https://smoothieware.org" target="_blank" rel="noopener noreferrer">Smoothieware</a>
);

const marlinLink = (
  <a href="https://marlinfw.org" target="_blank" rel="noopener noreferrer">Marlin</a>
);

const octoLink = (
  <a href="https://octoprint.org" target="_blank" rel="noopener noreferrer">OctoPrint</a>
);

export default function Devices() {
  return (
    <Layout
      title={translate({
        id: 'page.devices.layout.title',
        message: 'Supported Devices',
      })}
      description={translate({
        id: 'page.devices.layout.description',
        message: 'Browse the list of laser cutters, engravers, and controllers supported by Rayforge — including GRBL, Smoothieware, Ruida, and OctoPrint-based machines.',
      })}
    >
      <main className={styles.pageWrapper}>
        <section className={styles.hero}>
          <div className={styles.heroInner}>
            <h1 className={styles.heroTitle}>
              <Translate id="page.devices.hero.title">
                Supported Devices
              </Translate>
            </h1>
            <p className={styles.heroSubtitle}>
              <Translate id="page.devices.hero.subtitle">
                Rayforge works with a wide range of laser cutters, engravers,
                and controllers. Below are the built-in device profiles — pick
                one that matches your machine or use a generic profile for
                custom builds.
              </Translate>
            </p>
            <div className={styles.heroCtas}>
              <Link
                to="/docs/getting-started/installation"
                className={`rfButton rfButtonOrange ${styles.heroCtaButton}`}
              >
                <Icon path={mdiLaserPointer} size={0.9} />
                <span>
                  <Translate id="page.devices.hero.getStarted">
                    Get Started
                  </Translate>
                </span>
              </Link>
              <Link
                to="/docs/machine/config-wizard"
                className={`rfButton rfButtonPurple ${styles.heroCtaButton}`}
              >
                <span>
                  <Translate id="page.devices.hero.configGuide">
                    Configuration Guide
                  </Translate>
                </span>
              </Link>
            </div>
          </div>
        </section>

        {sections.map((section) => (
          <section key={section.groupName} className={styles.section}>
            <div className={styles.sectionInner}>
              <h2 className={styles.sectionTitle}>
                <Icon
                  path={section.label.icon}
                  size={1.1}
                  className={styles.sectionTitleIcon}
                />
                {section.label.label}
              </h2>

              {section.cards.length > 0 && (
                <div className={styles.cardGrid}>
                  {section.cards.map((device) => {
                    const affiliate = device.affiliate;
                    return (
                      <div key={device.id} className={styles.card}>
                        <a
                          href={affiliate.href}
                          target="_blank"
                          rel="noopener noreferrer"
                          className={styles.cardImageLink}
                        >
                          <img
                            src={affiliate.img}
                            alt={device.name}
                            className={styles.cardImage}
                          />
                        </a>
                        <div className={styles.cardBody}>
                          <h3 className={styles.cardTitle}>{device.name}</h3>
                          <p className={styles.cardDescription}>
                            {translate({id: `page.devices.device.${device.id}.description`, message: device.description})}
                          </p>
                        </div>
                        <a
                          href={affiliate.href}
                          className={styles.cardAffiliate}
                          target="_blank"
                          rel="noopener noreferrer"
                        >
                          <Icon path={mdiCartOutline} size={0.8} />
                          <span>
                            <Translate
                              id="page.devices.card.shop"
                              values={{ shop: affiliate.shop }}
                            >
                              {'Shop on {shop}'}
                            </Translate>
                          </span>
                        </a>
                      </div>
                    );
                  })}
                </div>
              )}

              {section.list.length > 0 && section.groupName !== 'Ruida' && (
                <ul className={styles.otherList}>
                  {section.list.map((device) => (
                    <li key={device.id} className={styles.otherItem}>
                      <strong>{device.name}</strong>
                        <span className={styles.otherDesc}>
                          {translate({id: `page.devices.device.${device.id}.description`, message: device.description})}
                        </span>
                    </li>
                  ))}
                </ul>
              )}

              {section.groupName === 'Grbl' && (
                <p className={`${styles.note} ${styles.noteInfo}`}>
                  <Icon path={mdiInformationOutline} size={1} className={`${styles.noteIcon} ${styles.noteIconInfo}`} />
                  <span>
                    <Translate
                      id="page.devices.note.grbl"
                      values={{ grblLink }}
                    >
                      {'Any {grblLink}-based device is supported, even without a built-in profile.'}
                    </Translate>
                  </span>
                </p>
              )}

              {section.groupName === 'Smoothieware' && (
                <p className={`${styles.note} ${styles.noteInfo}`}>
                  <Icon path={mdiInformationOutline} size={1} className={`${styles.noteIcon} ${styles.noteIconInfo}`} />
                  <span>
                    <Translate
                      id="page.devices.note.smoothieware"
                      values={{ smoothieLink }}
                    >
                      {'Other {smoothieLink}-based devices should also be supported.'}
                    </Translate>
                  </span>
                </p>
              )}

              {section.groupName === 'Marlin' && (
                <p className={`${styles.note} ${styles.noteInfo}`}>
                  <Icon path={mdiInformationOutline} size={1} className={`${styles.noteIcon} ${styles.noteIconInfo}`} />
                  <span>
                    <Translate
                      id="page.devices.note.marlin"
                      values={{
                        marlinLink,
                        marlinSerial: <strong>Marlin (Serial)</strong>,
                        configWizard: (
                          <Link to="/docs/machine/config-wizard">
                            configuration wizard
                          </Link>
                        ),
                      }}
                    >
                      {'Rayforge supports {marlinLink} firmware via the Marlin Serial driver. Select {marlinSerial} in the {configWizard} to connect your Marlin-based machine.'}
                    </Translate>
                  </span>
                </p>
              )}

              {section.groupName === 'OctoPrint' && (
                <p className={`${styles.note} ${styles.noteInfo}`}>
                  <Icon path={mdiInformationOutline} size={1} className={`${styles.noteIcon} ${styles.noteIconInfo}`} />
                  <span>
                    <Translate
                      id="page.devices.note.octoprint"
                      values={{ octoLink }}
                    >
                      {'Connect your laser through an {octoLink} server.'}
                    </Translate>
                  </span>
                </p>
              )}

              {section.groupName === 'Ruida' && (
                <p className={`${styles.note} ${styles.noteWarning}`}>
                  <Icon path={mdiAlertOutline} size={1} className={`${styles.noteIcon} ${styles.noteIconWarning}`} />
                  <span>
                    <Translate
                      id="page.devices.note.ruida"
                      values={{
                        genericRuida: <strong>Generic Ruida</strong>,
                      }}
                    >
                      {'Ruida support is experimental. For Ruida-based machines, try the {genericRuida} profile.'}
                    </Translate>
                  </span>
                </p>
              )}
            </div>
          </section>
        ))}

        <section className={styles.section}>
          <div className={styles.sectionInner}>
            <h2 className={styles.sectionTitle}>
              <Translate id="page.devices.generic.heading">
                Generic Profiles
              </Translate>
            </h2>
            <p className={styles.lead}>
              <Translate
                id="page.devices.generic.text"
                values={{
                  grblGeneric: <strong>Generic GRBL</strong>,
                  ruidaGeneric: <strong>Generic Ruida</strong>,
                  smoothieGeneric: <strong>Generic Smoothieware</strong>,
                  firmwareLink: (
                    <Link to="/docs/reference/firmware">
                      supported firmware
                    </Link>
                  ),
                  configWizard: (
                    <Link to="/docs/machine/config-wizard">
                      configuration wizard
                    </Link>
                  ),
                }}
              >
                {"If your device isn't listed above, try one of the generic profiles — {grblGeneric}, {ruidaGeneric}, {smoothieGeneric}. They work with any {firmwareLink} and can be configured through the {configWizard}."}
              </Translate>
            </p>
          </div>
        </section>

        <section className={styles.section}>
          <div className={styles.sectionInner}>
            <h2 className={styles.sectionTitle}>
              <Translate id="page.devices.missing.heading">
                Missing a Device?
              </Translate>
            </h2>
            <p className={styles.lead}>
              <Translate
                id="page.devices.missing.text"
                values={{
                  profilesLink: (
                    <Link to="/docs/developer/addon-overview">
                      installable device profiles
                    </Link>
                  ),
                  devDocs: (
                    <Link to="/docs/developer/getting-started">
                      developer documentation
                    </Link>
                  ),
                  ghIssues: (
                    <a
                      href="https://github.com/barebaric/rayforge/issues/new"
                      target="_blank"
                      rel="noopener noreferrer"
                    >
                      open an issue
                    </a>
                  ),
                }}
              >
                {'Rayforge uses {profilesLink} — new machines can be added without updating the application itself. If you\'d like to contribute a profile for your device, check the {devDocs} or {ghIssues}.'}
              </Translate>
            </p>
          </div>
        </section>
      </main>
    </Layout>
  );
}
