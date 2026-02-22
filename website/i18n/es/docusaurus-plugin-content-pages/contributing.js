import React from 'react';
import Layout from '@theme/Layout';
import Link from '@docusaurus/Link';
import styles from '@site/src/pages/contributing.module.css';

export default function Contributing() {
  return (
    <Layout
      title="Contribuir"
      description="Aprende cómo contribuir a Rayforge"
    >
      <div className="container container--fluid margin-vert--lg">
        <div className="row">
          <div className="col col--8 col--offset-2">
            <h1>Contribuir a Rayforge</h1>

            <div className={styles.supportSection}>
              <h2>Apoya el Proyecto</h2>
              <a href="https://www.patreon.com/c/knipknap">
                <img
                  src="https://c5.patreon.com/external/logo/become_a_patron_button.png"
                  alt="Convertirse en Mecenas"
                  height="55"
                />
              </a>
            </div>

            <h2>Comunidad y Soporte</h2>

            <ul>
              <li>
                <strong>Reportar Problemas</strong>:{' '}
                <a href="https://github.com/barebaric/rayforge/issues">
                  GitHub Issues
                </a>
              </li>
              <li>
                <strong>Código Fuente</strong>:{' '}
                <a href="https://github.com/barebaric/rayforge">
                  Repositorio de GitHub
                </a>
              </li>
            </ul>

            <p>
              ¡Damos la bienvenida a contribuciones de todo tipo! Ya sea que estés corrigiendo
              errores, añadiendo funciones, mejorando la documentación o ayudando con el
              empaquetado, tus contribuciones hacen que Rayforge sea mejor para todos.
            </p>

            <h2>Formas de Contribuir</h2>

            <h3>Reportar Errores</h3>

            <p>¿Encontraste un error? Ayúdanos a corregirlo:</p>

            <ol>
              <li>
                Revisa{' '}
                <a href="https://github.com/barebaric/rayforge/issues">
                  los problemas existentes
                </a>{' '}
                para evitar duplicados
              </li>
              <li>
                Crea un{' '}
                <a href="https://github.com/barebaric/rayforge/issues/new">
                  nuevo problema
                </a>{' '}
                con:
                <ul>
                  <li>Descripción clara del problema</li>
                  <li>Pasos para reproducirlo</li>
                  <li>Comportamiento esperado vs. real</li>
                  <li>Información del sistema (SO, versión de Rayforge)</li>
                  <li>Capturas de pantalla o mensajes de error si aplica</li>
                </ul>
              </li>
            </ol>

            <h3>Sugerir Funciones</h3>

            <p>¿Tienes una idea para una nueva función?</p>

            <ol>
              <li>
                Revisa{' '}
                <a href="https://github.com/barebaric/rayforge/issues?q=is%3Aissue+label%3Aenhancement">
                  las solicitudes de funciones existentes
                </a>
              </li>
              <li>
                Abre un problema de solicitud de función con:
                <ul>
                  <li>Descripción de la función</li>
                  <li>Caso de uso y beneficios</li>
                  <li>Posible enfoque de implementación (opcional)</li>
                </ul>
              </li>
            </ol>

            <h3>Enviar Código</h3>

            <p>
              Para información detallada sobre cómo enviar contribuciones de código,
              por favor consulta la guía{' '}
              <Link to="/docs/developer/getting-started">
                Documentación para Desarrolladores - Primeros Pasos
              </Link>{' '}.
            </p>

            <h3>Mejorar la Documentación</h3>

            <p>Las contribuciones a la documentación son muy valoradas:</p>

            <ul>
              <li>Corregir erratas o explicaciones poco claras</li>
              <li>Añadir ejemplos y capturas de pantalla</li>
              <li>Mejorar la organización</li>
              <li>Traducir a otros idiomas</li>
            </ul>

            <p>
              Puedes hacer clic en el botón "editar esta página" en cualquier página de documentación
              y luego enviar PRs de la misma manera que las contribuciones de código.
            </p>

            <h2>Sobre Esta Documentación</h2>

            <p>
              Esta documentación está diseñada para los usuarios finales de Rayforge. Si
              estás buscando documentación para desarrolladores, consulta la guía{' '}
              <Link to="/docs/developer/getting-started">
                Documentación para Desarrolladores
              </Link>{' '}.
            </p>
          </div>
        </div>
      </div>
    </Layout>
  );
}
