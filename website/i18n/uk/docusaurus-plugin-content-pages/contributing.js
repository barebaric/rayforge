import React from 'react';
import Layout from '@theme/Layout';
import Link from '@docusaurus/Link';
import styles from '@site/src/pages/contributing.module.css';

export default function Contributing() {
  return (
    <Layout
      title="Участь у проекті"
      description="Дізнайтеся, як взяти участь у розробці Rayforge"
    >
      <div className="container container--fluid margin-vert--lg">
        <div className="row">
          <div className="col col--8 col--offset-2">
            <h1>Участь у розробці Rayforge</h1>

            <div className={styles.supportSection}>
              <h2>Підтримати проект</h2>
              <a href="https://www.patreon.com/c/knipknap">
                <img
                  src="https://c5.patreon.com/external/logo/become_a_patron_button.png"
                  alt="Стати Patreon"
                  height="55"
                />
              </a>
            </div>

            <h2>Спільнота та підтримка</h2>

            <ul>
              <li>
                <strong>Повідомити про проблеми</strong>:{' '}
                <a href="https://github.com/barebaric/rayforge/issues">
                  GitHub Issues
                </a>
              </li>
              <li>
                <strong>Вихідний код</strong>:{' '}
                <a href="https://github.com/barebaric/rayforge">
                  Репозиторій GitHub
                </a>
              </li>
            </ul>

            <p>
              Ми вітаємо внесок будь-якого роду! Незалежно від того, чи виправляєте ви помилки, 
              додаєте функції, покращуєте документацію або допомагаєте з пакуванням — 
              ваш внесок робить Rayforge кращим для всіх.
            </p>

            <h2>Способи взяти участь</h2>

            <h3>Повідомити про помилки</h3>

            <p>Знайшли помилку? Допоможіть нам її виправити:</p>

            <ol>
              <li>
                Перевірте{' '}
                <a href="https://github.com/barebaric/rayforge/issues">
                  наявні issues
                </a>{' '}
                щоб уникнути дублювання
              </li>
              <li>
                Створіть{' '}
                <a href="https://github.com/barebaric/rayforge/issues/new">
                  нове issue
                </a>{' '}
                з:
                <ul>
                  <li>Чітким описом проблеми</li>
                  <li>Кроками для відтворення</li>
                  <li>Очікувана та фактична поведінка</li>
                  <li>Інформація про систему (ОС, версія Rayforge)</li>
                  <li>Знімки екрана або повідомлення про помилки, якщо застосовно</li>
                </ul>
              </li>
            </ol>

            <h3>Запропонувати функції</h3>

            <p>Маєте ідею для нової функції?</p>

            <ol>
              <li>
                Перевірте{' '}
                <a href="https://github.com/barebaric/rayforge/issues?q=is%3Aissue+label%3Aenhancement">
                  наявні запити на функції
                </a>
              </li>
              <li>
                Створіть запит на функцію з:
                <ul>
                  <li>Опис функції</li>
                  <li>Варіант використання та переваги</li>
                  <li>Можливий підхід до реалізації (необов'язково)</li>
                </ul>
              </li>
            </ol>

            <h3>Надіслати код</h3>

            <p>
              Детальну інформацію про надсилання коду можна знайти в{' '}
              <Link to="/docs/developer/getting-started">
                документації для розробників - Початок роботи
              </Link>{' '}
              посібнику.
            </p>

            <h3>Покращити документацію</h3>

            <p>Внески до документації дуже вітаються:</p>

            <ul>
              <li>Виправлення помилок або незрозумілих пояснень</li>
              <li>Додавання прикладів та знімків екрана</li>
              <li>Покращення структури</li>
              <li>Переклад іншими мовами</li>
            </ul>

            <p>
              Ви можете натиснути кнопку «Редагувати цю сторінку» на будь-якій сторінці 
              документації, а потім надіслати PR так само, як і внески коду.
            </p>

            <h2>Про цю документацію</h2>

            <p>
              Ця документація призначена для кінцевих користувачів Rayforge.
              Якщо ви шукаєте документацію для розробників, вона знаходиться в{' '}
              <Link to="/docs/developer/getting-started">
                документації для розробників
              </Link>{' '}
              посібнику.
            </p>
          </div>
        </div>
      </div>
    </Layout>
  );
}
