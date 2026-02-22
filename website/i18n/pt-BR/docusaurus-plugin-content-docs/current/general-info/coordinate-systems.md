# Sistemas de Coordenadas

Entender como o Rayforge lida com sistemas de coordenadas é essencial para posicionar seu trabalho corretamente.

## Sistema de Coordenadas de Trabalho (WCS) vs Coordenadas de Máquina

O Rayforge usa dois sistemas principais de coordenadas:

### Sistema de Coordenadas de Trabalho (WCS)

O WCS é o sistema de coordenadas do seu trabalho. Quando você posiciona um design em (50, 100) na tela, essas são coordenadas WCS.

- **Origem**: Definida por você (padrão é G54)
- **Propósito**: Design e posicionamento do trabalho
- **Múltiplos sistemas**: G54-G59 disponíveis para diferentes configurações

### Coordenadas de Máquina

Coordenadas de máquina são posições absolutas relativas à posição home da máquina.

- **Origem**: Home da máquina (0,0,0) - fixado pelo hardware
- **Propósito**: Posicionamento físico na mesa
- **Fixo**: Não pode ser alterado por software

**Relação**: Os deslocamentos WCS definem como suas coordenadas de trabalho mapeiam para coordenadas de máquina. Se o deslocamento G54 é (100, 50, 0), então seu design em WCS (0, 0) corta na posição de máquina (100, 50).

## Configurando Coordenadas no Rayforge

### Definindo a Origem WCS

Para posicionar seu trabalho na máquina:

1. **Faça home da máquina** primeiro (comando `$H` ou botão Home)
2. **Mova a cabeça do laser** para sua origem de trabalho desejada
3. **Defina zero WCS** usando o Painel de Controle:
   - Clique "Zero X" para definir o X atual como origem
   - Clique "Zero Y" para definir o Y atual como origem
4. Seu trabalho agora vai começar desta posição

### Selecionando um WCS

O Rayforge suporta sistemas de coordenadas de trabalho G54-G59:

| Sistema | Caso de Uso |
|--------|----------|
| G54 | Padrão, área de trabalho principal |
| G55-G59 | Posições de fixação adicionais |

Selecione o WCS ativo no Painel de Controle. Cada sistema armazena seu próprio deslocamento da origem da máquina.

### Direção do Eixo Y

Algumas máquinas têm Y aumentando para baixo em vez de para cima. Configure isso em:

**Configurações → Máquina → Hardware → Eixos**

Se seus trabalhos saem espelhados verticalmente, alterne a configuração de direção do eixo Y.

## Problemas Comuns

### Trabalho na Posição Errada

- **Verifique o deslocamento WCS**: Envie `G10 L20 P1` para ver o deslocamento G54
- **Verifique o homing**: A máquina deve fazer home para posicionamento consistente
- **Verifique a direção do eixo Y**: Pode estar invertido

### Coordenadas Variam Entre Trabalhos

- **Sempre faça home antes dos trabalhos**: Estabelece referência consistente
- **Verifique deslocamentos G92**: Limpe com o comando `G92.1`

---

## Páginas Relacionadas

- [Sistemas de Coordenadas de Trabalho (WCS)](work-coordinate-systems) - Gerenciando WCS no Rayforge
- [Painel de Controle](../ui/control-panel) - Controles de movimento e botões WCS
- [Exportando G-code](../files/exporting) - Opções de posicionamento de trabalho
