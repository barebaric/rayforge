# Gravação

As operações de gravação preenchem áreas com linhas de varredura raster, suportando múltiplos modos para diferentes efeitos de gravação. De fotos em escala de cinza suaves a efeitos de relevo 3D, escolha o modo que melhor se adapta ao seu design e material.

## Visão Geral

Operações de gravação:

- Preenchem formas fechadas com linhas de varredura
- Suportam múltiplos modos de gravação para diferentes efeitos
- Funcionam tanto com formas vetoriais quanto imagens bitmap
- Usam varredura bidirecional para velocidade
- Criam marcas permanentes em muitos materiais

## Modos de Gravação

### Modo de Potência Variável

O modo de Potência Variável varia a potência do laser continuamente com base no brilho da imagem, criando gravação em escala de cinza suave com transições graduais.

**Melhor Para:**

- Fotos e imagens em escala de cinza suaves
- Gradientes e transições naturais
- Retratos e obras de arte
- Gravação em madeira e couro

**Recursos Principais:**

- Modulação contínua de potência
- Controle de potência mín/máx
- Gradientes suaves
- Melhor qualidade tonal que pontilhamento

### Modo de Potência Constante

O modo de Potência Constante grava em potência total, com um limiar determinando quais pixels são gravados. Isso cria resultados limpos em preto/branco.

**Melhor Para:**

- Texto e logos
- Gráficos de alto contraste
- Gravações limpas em preto/branco
- Formas e padrões simples

**Recursos Principais:**

- Gravação baseada em limiar
- Saída de potência consistente
- Mais rápido que modo de potência variável
- Bordas limpas

### Modo Pontilhado

O modo Pontilhado converte imagens em escala de cinza para padrões binários usando algoritmos de pontilhamento, permitindo gravação de fotos de alta qualidade com melhor reprodução tonal que métodos baseados apenas em limiar.

**Melhor Para:**

- Gravando fotografias em madeira ou couro
- Criando obras de arte estilo meio-tom
- Imagens com gradientes suaves
- Quando gravação raster padrão não captura detalhe suficiente

**Recursos Principais:**

- Múltiplas escolhas de algoritmos de pontilhamento
- Melhor preservação de detalhes
- Tons contínuos percebidos
- Ideal para fotografias

### Modo Múltiplas Profundidades

O modo Múltiplas Profundidades cria efeitos de relevo 3D variando a potência do laser com base no brilho da imagem, com múltiplas passagens para escavação mais profunda.

**Melhor Para:**

- Criando retratos e obras de arte 3D
- Mapas de terreno e topográficos
- Litofanias (imagens 3D transmissoras de luz)
- Logos e designs em relevo
- Esculturas em relevo

**Recursos Principais:**

- Mapeamento de profundidade a partir do brilho da imagem
- Profundidade mín/máx configurável
- Gradientes suaves
- Múltiplas passagens para gravação mais profunda
- Degraus Z entre passagens

## Quando Usar Gravação

Use operações de gravação para:

- Gravar texto e logos
- Criar imagens e fotos em madeira/couro
- Preencher áreas sólidas com textura
- Marcar peças e produtos
- Criar efeitos de relevo 3D
- Obras de arte estilo meio-tom

**Não use gravação para:**

- Cortar através do material (use [Contorno](contour) em vez disso)
- Contornos precisos (raster cria áreas preenchidas)
- Trabalho de linhas finas (vetores são mais limpos)

## Criando uma Operação de Gravação

### Passo 1: Preparar Conteúdo

Gravação funciona com:

- **Formas vetoriais** - Preenchidas com linhas de varredura
- **Texto** - Convertido para caminhos preenchidos
- **Imagens** - Convertidas para escala de cinza e gravadas

### Passo 2: Adicionar Operação de Gravação

- **Menu:** Operações → Adicionar Gravação
- **Atalho:** <kbd>ctrl+shift+e</kbd>
- **Clique direito:** Menu de contexto → Adicionar Operação → Gravação

### Passo 3: Escolher Modo

Selecione o modo de gravação que melhor se adapta às suas necessidades:

- **Potência Variável** - Gravação em escala de cinza suave
- **Potência Constante** - Gravação limpa em preto/branco
- **Pontilhado** - Gravação de fotos de alta qualidade
- **Múltiplas Profundidades** - Efeitos de relevo 3D

### Passo 4: Configurar Definições

![Configurações de etapa de gravação](/screenshots/step-settings-engrave-general-variable.png)

## Configurações Comuns

### Potência e Velocidade

**Potência (%):**

- Intensidade do laser para gravação
- Menor potência para marcação mais leve
- Maior potência para gravação mais profunda

**Velocidade (mm/min):**

- Quão rápido o laser varre
- Mais rápido = mais claro, mais lento = mais escuro

### Intervalo de Linha

**Intervalo de Linha (mm):**

- Espaçamento entre linhas de varredura
- Menor = maior qualidade, tempo de trabalho mais longo
- Maior = mais rápido, linhas visíveis

| Intervalo | Qualidade | Velocidade   | Usar Para               |
| -------- | ------- | ------- | --------------------- |
| 0.05mm   | Mais Alta | Mais Lento | Fotos, detalhes finos   |
| 0.1mm    | Alta    | Médio  | Texto, logos, gráficos |
| 0.2mm    | Média  | Rápido    | Preenchimentos sólidos, texturas |
| 0.3mm+   | Baixa    | Mais Rápido | Rascunho, teste        |

**Recomendado:** 0.1mm para uso geral

:::tip Correspondência de Resolução
:::

Para imagens, o intervalo de linha deve corresponder ou exceder a resolução da imagem. Se sua imagem tem 10 pixels/mm (254 DPI), use intervalo de linha de 0.1mm ou menor.

### Direção de Varredura

**Ângulo de Varredura (graus):**

- Direção das linhas de varredura
- 0 = horizontal (esquerda para direita)
- 90 = vertical (cima para baixo)
- 45 = diagonal

**Por que mudar o ângulo?**

- Grão da madeira: Grave perpendicular ao grão para melhores resultados
- Orientação do padrão: Combine a estética do design
- Reduzir faixas: Ângulo diferente pode esconder imperfeições

**Varredura Bidirecional:**

- **Habilitado:** Laser grava em ambas direções (mais rápido)
- **Desabilitado:** Laser só grava da esquerda para direita (mais lento, mais consistente)

Para melhor qualidade, desabilite bidirecional. Para velocidade, habilite.

### Overscan

**Distância de Overscan (mm):**

- Quão longe além do design o laser viaja antes de virar
- Permite o laser atingir velocidade total antes de entrar no design
- Previne marcas de queimadura nos inícios/fins das linhas

**Valores típicos:**

- 2-5mm para a maioria dos trabalhos
- Maior para altas velocidades

Veja [Overscan](../overscan) para detalhes.

## Configurações Específicas do Modo

### Configurações do Modo Potência Variável

![Configurações do modo Potência Variável](/screenshots/step-settings-engrave-general-variable.png)

**Potência Mín (%):**

- Potência do laser para áreas mais claras (pixels brancos)
- Geralmente 0-20%
- Defina mais alto para evitar áreas muito rasas

**Potência Máx (%):**

- Potência do laser para áreas mais escuras (pixels pretos)
- Geralmente 40-80% dependendo do material
- Menor = relevo sutil, maior = profundidade dramática

**Exemplos de Faixa de Potência:**

| Mín | Máx | Efeito                |
| --- | --- | --------------------- |
| 0%  | 40% | Relevo sutil, claro  |
| 10% | 60% | Profundidade média, seguro    |
| 20% | 80% | Relevo profundo, dramático |

**Inverter:**

- **Desligado** (padrão): Branco = raso, Preto = profundo
- **Ligado**: Branco = profundo, Preto = raso

Use inverter para litofanias (áreas claras devem ser finas) ou embossing (áreas elevadas).

**Faixa de Brilho:**

Controla como os valores de brilho da imagem são mapeados para potência do laser. O histograma mostra a distribuição de valores de brilho na sua imagem.

- **Níveis Automáticos** (padrão): Ajusta automaticamente os pontos preto e branco com base no conteúdo da imagem. Valores abaixo do ponto preto são tratados como preto, valores acima do ponto branco são tratados como branco. Isso estica o contraste da imagem para usar a faixa completa de potência.
- **Modo Manual**: Desabilite Níveis Automáticos para definir manualmente os pontos preto e branco arrastando os marcadores no histograma.

Isso é particularmente útil para:
- Imagens de baixo contraste que precisam de aumento de contraste
- Imagens com faixa tonal limitada
- Garantir resultados consistentes entre diferentes imagens de origem

### Configurações do Modo Potência Constante

![Configurações do modo Potência Constante](/screenshots/step-settings-engrave-general-constant_power.png)

**Limiar (0-255):**

- Corte de brilho para separação preto/branco
- Menor = mais preto gravado
- Maior = mais branco gravado

**Valores típicos:**

- 128 (limiar de cinza 50%)
- Ajuste com base no contraste da imagem

### Configurações do Modo Pontilhado

![Configurações do modo Pontilhado](/screenshots/step-settings-engrave-general-dither.png)

**Algoritmo de Pontilhamento:**

Escolha o algoritmo que melhor se adapta à sua imagem e material:

| Algoritmo       | Qualidade | Velocidade   | Melhor Para                            |
| --------------- | ------- | ------- | ----------------------------------- |
| Floyd-Steinberg | Mais Alta | Mais Lento | Fotos, retratos, gradientes suaves |
| Bayer 2x2       | Baixa     | Mais Rápido | Efeito meio-tom grosseiro              |
| Bayer 4x4       | Média  | Rápido    | Meio-tom balanceado                   |
| Bayer 8x8       | Alta    | Médio  | Detalhe fino, padrões sutis        |

**Floyd-Steinberg** é padrão e recomendado para a maioria das gravações de fotos. Usa difusão de erro para distribuir erros de quantização para pixels vizinhos, criando resultados de aparência natural.

**Pontilhamento Bayer** cria padrões regulares que podem produzir efeitos artísticos lembrando impressão tradicional de meio-tom.

### Configurações do Modo Múltiplas Profundidades

![Configurações do modo Múltiplas Profundidades](/screenshots/step-settings-engrave-general-multi_pass.png)

**Número de Níveis de Profundidade:**

- Número de níveis de profundidade discretos
- Mais níveis = gradientes mais suaves
- Típico: 5-10 níveis

**Degrau Z por Nível (mm):**

- Quão baixo descer entre passagens de profundidade
- Cria profundidade total mais profunda com múltiplas passagens
- Típico: 0.1-0.5mm

**Rotacionar Ângulo Por Passagem:**

- Graus para rotacionar cada passagem sucessiva
- Cria efeito 3D estilo malha cruzada
- Típico: 0-45 graus

**Inverter:**

- **Habilitado:** Branco = profundo, Preto = raso
- **Desabilitado:** Preto = profundo, Branco = raso

Use inverter para litofanias (áreas claras devem ser finas) ou embossing (áreas elevadas).

## Dicas e Melhores Práticas

![Configurações de pós-processamento de gravação](/screenshots/step-settings-engrave-post.png)

### Seleção de Material

**Melhores materiais para gravação:**

- Madeira (variações naturais criam resultados belos)
- Couro (queima para marrom escuro/preto)
- Alumínio anodizado (remove revestimento, revela metal)
- Metais revestidos (remove camada de revestimento)
- Alguns plásticos (teste primeiro!)

**Materiais desafiadores:**

- Acrílico transparente (não mostra gravação bem)
- Metais sem revestimento (requer compostos de marcação especiais)
- Vidro (requer configurações/revestimentos especiais)

### Configurações de Qualidade

**Para melhor qualidade:**

- Use intervalo de linha menor (0.05-0.1mm)
- Desabilite varredura bidirecional
- Aumente overscan (3-5mm)
- Use potência menor, múltiplas passagens
- Certifique-se de que o material está plano e fixado

**Para gravação mais rápida:**

- Use intervalo de linha maior (0.15-0.2mm)
- Habilite varredura bidirecional
- Overscan mínimo (1-2mm)
- Passagem única em potência maior

### Problemas Comuns

**Marcas de queimadura nos fins das linhas:**

- Aumente a distância de overscan
- Verifique configurações de aceleração
- Reduza a potência ligeiramente

**Linhas de varredura visíveis:**

- Diminua o intervalo de linha
- Reduza potência (sobre-queima cria lacunas)
- Verifique se o material está plano

**Gravação irregular:**

- Certifique-se de que o material está plano
- Verifique consistência do foco
- Verifique estabilidade da potência do laser
- Limpe a lente do laser

**Faixas (listras escuras/claras):**

- Desabilite varredura bidirecional
- Verifique tensão das correias
- Reduza velocidade
- Tente ângulo de varredura diferente

## Solução de Problemas

### Gravação muito clara

- **Aumente:** Configuração de potência
- **Diminua:** Configuração de velocidade
- **Verifique:** Foco está correto
- **Tente:** Múltiplas passagens

### Gravação muito escura/queimando

- **Diminua:** Configuração de potência
- **Aumente:** Configuração de velocidade
- **Aumente:** Intervalo de linha
- **Verifique:** Material é apropriado

### Escuridão inconsistente

- **Verifique:** Material está plano
- **Verifique:** Distância do foco é consistente
- **Verifique:** Feixe do laser está limpo
- **Teste:** Área diferente do material (grão varia)

### Imagem parece pixelada

- **Diminua:** Intervalo de linha
- **Verifique:** Resolução da imagem de origem
- **Tente:** Intervalo de linha menor (0.05mm)
- **Verifique:** Imagem não está sendo ampliada

### Linhas de varredura visíveis

- **Diminua:** Intervalo de linha
- **Reduza:** Potência (sobre-queima cria lacunas)
- **Tente:** Ângulo de varredura diferente
- **Certifique-se:** Superfície do material está lisa

## Tópicos Relacionados

- **[Corte de Contorno](contour)** - Cortando contornos e formas
- **[Overscan](../overscan)** - Melhorando qualidade de gravação
- **[Grade de Teste de Material](material-test-grid)** - Encontrando configurações ideais
- **[Fluxo de Trabalho Multi-Camadas](../multi-layer)** - Combinando gravação com outras operações
