import React from 'react';
import Layout from '@theme/Layout';
import Link from '@docusaurus/Link';
import styles from '@site/src/pages/contributing.module.css';

export default function Contributing() {
  return (
    <Layout
      title="Mitwirken"
      description="Erfahre, wie du bei Rayforge mitwirken kannst"
    >
      <div className="container container--fluid margin-vert--lg">
        <div className="row">
          <div className="col col--8 col--offset-2">
            <h1>Bei Rayforge mitwirken</h1>

            <div className={styles.supportSection}>
              <h2>Das Projekt unterstützen</h2>
              <a href="https://www.patreon.com/c/knipknap">
                <img
                  src="https://c5.patreon.com/external/logo/become_a_patron_button.png"
                  alt="Patreon werden"
                  height="55"
                />
              </a>
            </div>

            <h2>Community & Unterstützung</h2>

            <ul>
              <li>
                <strong>Probleme melden</strong>:{' '}
                <a href="https://github.com/barebaric/rayforge/issues">
                  GitHub Issues
                </a>
              </li>
              <li>
                <strong>Quellcode</strong>:{' '}
                <a href="https://github.com/barebaric/rayforge">
                  GitHub Repository
                </a>
              </li>
            </ul>

            <p>
              Wir freuen uns über Beiträge aller Art! Egal ob du Fehler behebst,
              Funktionen hinzufügst, die Dokumentation verbesserst oder bei der
              Verpackung hilfst – deine Beiträge machen Rayforge für alle besser.
            </p>

            <h2>Möglichkeiten mitzuwirken</h2>

            <h3>Fehler melden</h3>

            <p>Einen Fehler gefunden? Hilf uns, ihn zu beheben:</p>

            <ol>
              <li>
                Prüfe{' '}
                <a href="https://github.com/barebaric/rayforge/issues">
                  bestehende Issues
                </a>{' '}
                um Duplikate zu vermeiden
              </li>
              <li>
                Erstelle ein{' '}
                <a href="https://github.com/barebaric/rayforge/issues/new">
                  neues Issue
                </a>{' '}
                mit:
                <ul>
                  <li>Klare Beschreibung des Problems</li>
                  <li>Schritte zum Reproduzieren</li>
                  <li>Erwartetes vs. tatsächliches Verhalten</li>
                  <li>Systeminformationen (Betriebssystem, Rayforge-Version)</li>
                  <li>Screenshots oder Fehlermeldungen falls zutreffend</li>
                </ul>
              </li>
            </ol>

            <h3>Funktionen vorschlagen</h3>

            <p>Hast du eine Idee für eine neue Funktion?</p>

            <ol>
              <li>
                Prüfe{' '}
                <a href="https://github.com/barebaric/rayforge/issues?q=is%3Aissue+label%3Aenhancement">
                  bestehende Funktionswünsche
                </a>
              </li>
              <li>
                Eröffne einen Funktionswunsch mit:
                <ul>
                  <li>Beschreibung der Funktion</li>
                  <li>Anwendungsfall und Vorteile</li>
                  <li>Möglicher Implementierungsansatz (optional)</li>
                </ul>
              </li>
            </ol>

            <h3>Code einreichen</h3>

            <p>
              Ausführliche Informationen zum Einreichen von Code-Beiträgen
              findest du im{' '}
              <Link to="/docs/developer/getting-started">
                Entwickler-Dokumentation - Erste Schritte
              </Link>{' '}
              Leitfaden.
            </p>

            <h3>Dokumentation verbessern</h3>

            <p>Beiträge zur Dokumentation sind sehr willkommen:</p>

            <ul>
              <li>Tippfehler oder unklare Erklärungen korrigieren</li>
              <li>Beispiele und Screenshots hinzufügen</li>
              <li>Struktur verbessern</li>
              <li>In andere Sprachen übersetzen</li>
            </ul>

            <p>
              Du kannst auf jeder Dokumentationsseite auf die Schaltfläche
              „Diese Seite bearbeiten" klicken und dann PRs auf die gleiche
              Weise wie Code-Beiträge einreichen.
            </p>

            <h2>Über diese Dokumentation</h2>

            <p>
              Diese Dokumentation ist für Endanwender von Rayforge konzipiert.
              Wenn du nach Entwickler-Dokumentation suchst, findest du diese im{' '}
              <Link to="/docs/developer/getting-started">
                Entwickler-Dokumentation
              </Link>{' '}
              Leitfaden.
            </p>
          </div>
        </div>
      </div>
    </Layout>
  );
}
