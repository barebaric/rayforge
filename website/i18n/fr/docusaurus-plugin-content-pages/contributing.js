import React from 'react';
import Layout from '@theme/Layout';
import Link from '@docusaurus/Link';
import styles from '@site/src/pages/contributing.module.css';

export default function Contributing() {
  return (
    <Layout
      title="Contribuer"
      description="Apprendre comment contribuer à Rayforge"
    >
      <div className="container container--fluid margin-vert--lg">
        <div className="row">
          <div className="col col--8 col--offset-2">
            <h1>Contribuer à Rayforge</h1>

            <div className={styles.supportSection}>
              <h2>Soutenir le Projet</h2>
              <a href="https://www.patreon.com/c/knipknap">
                <img
                  src="https://c5.patreon.com/external/logo/become_a_patron_button.png"
                  alt="Devenir un Patron"
                  height="55"
                />
              </a>
            </div>

            <h2>Communauté et Support</h2>

            <ul>
              <li>
                <strong>Signaler des Problèmes</strong> :{' '}
                <a href="https://github.com/barebaric/rayforge/issues">
                  Issues GitHub
                </a>
              </li>
              <li>
                <strong>Code Source</strong> :{' '}
                <a href="https://github.com/barebaric/rayforge">
                  Dépôt GitHub
                </a>
              </li>
            </ul>

            <p>
              Nous accueillons les contributions de toutes sortes ! Que vous corrigiez
              des bugs, ajoutiez des fonctionnalités, amélioriez la documentation ou aidiez avec
              l'emballage, vos contributions rendent Rayforge meilleur pour tout le monde.
            </p>

            <h2>Manières de Contribuer</h2>

            <h3>Signaler des Bugs</h3>

            <p>Vous avez trouvé un bug ? Aidez-nous à le corriger :</p>

            <ol>
              <li>
                Consultez les{' '}
                <a href="https://github.com/barebaric/rayforge/issues">
                  problèmes existants
                </a>{' '}
                pour éviter les doublons
              </li>
              <li>
                Créez un{' '}
                <a href="https://github.com/barebaric/rayforge/issues/new">
                  nouveau problème
                </a>{' '}
                avec :
                <ul>
                  <li>Description claire du problème</li>
                  <li>Étapes pour reproduire</li>
                  <li>Comportement attendu vs. actuel</li>
                  <li>Informations système (OS, version de Rayforge)</li>
                  <li>Captures d'écran ou messages d'erreur si applicable</li>
                </ul>
              </li>
            </ol>

            <h3>Suggérer des Fonctionnalités</h3>

            <p>Vous avez une idée pour une nouvelle fonctionnalité ?</p>

            <ol>
              <li>
                Consultez les{' '}
                <a href="https://github.com/barebaric/rayforge/issues?q=is%3Aissue+label%3Aenhancement">
                  demandes de fonctionnalités existantes
                </a>
              </li>
              <li>
                Ouvrez un problème de demande de fonctionnalité avec :
                <ul>
                  <li>Description de la fonctionnalité</li>
                  <li>Cas d'utilisation et avantages</li>
                  <li>Approche d'implémentation possible (optionnel)</li>
                </ul>
              </li>
            </ol>

            <h3>Soumettre du Code</h3>

            <p>
              Pour des informations détaillées sur la soumission de contributions de code,
              veuillez consulter le guide{' '}
              <Link to="/docs/developer/getting-started">
                Documentation Développeur - Pour Commencer
              </Link>{' '}
              .
            </p>

            <h3>Améliorer la Documentation</h3>

            <p>Les contributions à la documentation sont très appréciées :</p>

            <ul>
              <li>Corriger les fautes de frappe ou les explications peu claires</li>
              <li>Ajouter des exemples et des captures d'écran</li>
              <li>Améliorer l'organisation</li>
              <li>Traduire dans d'autres langues</li>
            </ul>

            <p>
              Vous pouvez cliquer sur le bouton "modifier cette page" sur n'importe quelle page
              de documentation, puis soumettre des PR de la même manière que les contributions de code.
            </p>

            <h2>À Propos de Cette Documentation</h2>

            <p>
              Cette documentation est conçue pour les utilisateurs finaux de Rayforge. Si
              vous cherchez la documentation développeur, veuillez consulter le guide{' '}
              <Link to="/docs/developer/getting-started">
                Documentation Développeur
              </Link>{' '}
              .
            </p>
          </div>
        </div>
      </div>
    </Layout>
  );
}
