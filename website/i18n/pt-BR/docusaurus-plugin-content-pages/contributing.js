import React from 'react';
import Layout from '@theme/Layout';
import Link from '@docusaurus/Link';
import styles from '@site/src/pages/contributing.module.css';

export default function Contributing() {
  return (
    <Layout
      title="Contribuir"
      description="Aprenda como contribuir para o Rayforge"
    >
      <div className="container container--fluid margin-vert--lg">
        <div className="row">
          <div className="col col--8 col--offset-2">
            <h1>Contribuindo para o Rayforge</h1>

            <div className={styles.supportSection}>
              <h2>Apoie o Projeto</h2>
              <a href="https://www.patreon.com/c/knipknap">
                <img
                  src="https://c5.patreon.com/external/logo/become_a_patron_button.png"
                  alt="Torne-se um Patrono"
                  height="55"
                />
              </a>
            </div>

            <h2>Comunidade e Suporte</h2>

            <ul>
              <li>
                <strong>Reportar Problemas</strong>:{' '}
                <a href="https://github.com/barebaric/rayforge/issues">
                  Issues no GitHub
                </a>
              </li>
              <li>
                <strong>Código Fonte</strong>:{' '}
                <a href="https://github.com/barebaric/rayforge">
                  Repositório GitHub
                </a>
              </li>
            </ul>

            <p>
              Acolhemos contribuições de todos os tipos! Seja corrigindo bugs,
              adicionando recursos, melhorando a documentação ou ajudando com
              empacotamento, suas contribuições tornam o Rayforge melhor para todos.
            </p>

            <h2>Formas de Contribuir</h2>

            <h3>Reportar Bugs</h3>

            <p>Encontrou um bug? Ajude-nos a corrigir:</p>

            <ol>
              <li>
                Verifique os{' '}
                <a href="https://github.com/barebaric/rayforge/issues">
                  issues existentes
                </a>{' '}
                para evitar duplicatas
              </li>
              <li>
                Crie um{' '}
                <a href="https://github.com/barebaric/rayforge/issues/new">
                  novo issue
                </a>{' '}
                com:
                <ul>
                  <li>Descrição clara do problema</li>
                  <li>Passos para reproduzir</li>
                  <li>Comportamento esperado vs. atual</li>
                  <li>Informações do sistema (SO, versão do Rayforge)</li>
                  <li>Capturas de tela ou mensagens de erro, se aplicável</li>
                </ul>
              </li>
            </ol>

            <h3>Sugerir Recursos</h3>

            <p>Tem uma ideia para um novo recurso?</p>

            <ol>
              <li>
                Verifique as{' '}
                <a href="https://github.com/barebaric/rayforge/issues?q=is%3Aissue+label%3Aenhancement">
                  solicitações de recursos existentes
                </a>
              </li>
              <li>
                Abra um issue de solicitação de recurso com:
                <ul>
                  <li>Descrição do recurso</li>
                  <li>Caso de uso e benefícios</li>
                  <li>Abordagem possível de implementação (opcional)</li>
                </ul>
              </li>
            </ol>

            <h3>Enviar Código</h3>

            <p>
              Para informações detalhadas sobre o envio de contribuições de código,
              consulte o guia{' '}
              <Link to="/docs/developer/getting-started">
                Documentação para Desenvolvedores - Primeiros Passos
              </Link>{' '}
              .
            </p>

            <h3>Melhorar a Documentação</h3>

            <p>Contribuições de documentação são muito valorizadas:</p>

            <ul>
              <li>Corrigir erros de digitação ou explicações confusas</li>
              <li>Adicionar exemplos e capturas de tela</li>
              <li>Melhorar a organização</li>
              <li>Traduzir para outros idiomas</li>
            </ul>

            <p>
              Você pode clicar no botão "editar esta página" em qualquer página de documentação
              e depois enviar PRs da mesma forma que contribuições de código.
            </p>

            <h2>Sobre Esta Documentação</h2>

            <p>
              Esta documentação é projetada para usuários finais do Rayforge. Se
              você está procurando documentação para desenvolvedores, consulte o guia{' '}
              <Link to="/docs/developer/getting-started">
                Documentação para Desenvolvedores
              </Link>{' '}
              .
            </p>
          </div>
        </div>
      </div>
    </Layout>
  );
}
