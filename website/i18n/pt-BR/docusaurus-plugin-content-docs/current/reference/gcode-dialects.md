# Suporte a Dialeto G-code

O Rayforge suporta múltiplos dialetos G-code para trabalhar com diferentes
firmware de controlador.

## Dialetos Suportados

O Rayforge atualmente suporta estes dialetos G-code:

| Dialeto                                      | Firmware     | Uso Comum                                |
| -------------------------------------------- | ------------ | ---------------------------------------- |
| **Grbl (Compat)**                            | GRBL 1.1+    | Lasers de diodo, CNC hobby               |
| **Grbl (Compat, sem eixo Z)**                | GRBL 1.1+    | Cortadores a laser 2D sem Z              |
| **Grbl Raster**                              | GRBL 1.1+    | Otimizado para trabalho raster           |
| **GRBL Dinâmico (Sensível à Profundidade)**  | GRBL 1.1+    | Gravação a laser sensível à profundidade |
| **GRBL Dinâmico (sem eixo Z)**               | GRBL 1.1+    | Gravação a laser sensível à profundidade |
| **Mach4 (M67 Analog)**                       | Mach4        | Gravação raster de alta velocidade       |
| **Smoothieware**                             | Smoothieware | Cortadores a laser, CNC                  |
| **Marlin**                                   | Marlin 2.0+  | Impressoras 3D com laser                 |

:::note Dialetos Recomendados
:::

**Grbl (Compat)** é o dialeto mais testado e recomendado para aplicações de
laser padrão.

**Grbl Raster** é otimizado para gravação raster em controladores GRBL. Ele
mantém o laser no modo de potência dinâmica (M4) continuamente e omite comandos
de taxa de avanço redundantes, resultando em G-code mais suave e compacto.

**GRBL Dinâmico (Sensível à Profundidade)** é recomendado para gravação a laser
sensível à profundidade onde a potência varia durante os cortes (ex., gravação
de profundidade variável).

---

## Mach4 (M67 Analog)

O dialeto **Mach4 (M67 Analog)** é projetado para gravação raster de alta
velocidade com controladores Mach4. Ele usa o comando M67 com saída analógica
para controle preciso de potência do laser.

### Recursos Principais

- **Saída Analógica M67**: Usa `M67 E0 Q<0-255>` para potência do laser em vez
  de comandos S inline
- **Pressão de Buffer Reduzida**: Ao separar comandos de potência de comandos de
  movimento, o buffer do controlador é menos estressado durante operações de
  alta velocidade
- **Raster de Alta Velocidade**: Otimizado para operações de gravação raster
  rápidas

### Quando Usar

Use este dialeto quando:

- Você tem um controlador Mach4 com capacidade de saída analógica
- Você precisa de gravação raster de alta velocidade
- Seu controlador experimenta estouro de buffer com comandos S inline padrão

### Formato de Comando

O dialeto gera G-code como:

```gcode
M67 E0 Q127  ; Definir potência do laser para 50% (127/255)
G1 X100 Y200 F1000  ; Mover para posição
M67 E0 Q0    ; Desligar laser
```

---

## Criando um Dialeto Personalizado

Para criar um dialeto G-code personalizado baseado em um dialeto embutido:

1. Abra **Configurações de Máquina** → **Dialeto G-code**
2. Clique no ícone **Copiar** em um dialeto embutido para criar um novo dialeto
   personalizado
3. Edite as configurações do dialeto conforme necessário
4. Salve seu dialeto personalizado

Cada dialeto personalizado é uma cópia independente. Alterar um dialeto nunca
afeta os outros, então você pode experimentar livremente sem se preocupar em
danificar uma configuração existente. Dialetos personalizados são armazenados no
seu diretório de configuração e podem ser compartilhados.

### Configurações do Dialeto

Ao editar um dialeto personalizado, a página de Configurações oferece estas
opções:

**Modo Laser Contínuo** mantém o laser no modo de potência dinâmica (M4) ativo
durante todo o trabalho em vez de alternar M4/M5 entre segmentos. Isso é útil
para gravação raster onde o laser precisa permanecer ligado continuamente durante
as linhas de varredura.

**Taxa de Avanço Modal** omite o parâmetro de taxa de avanço (F) dos comandos de
movimento quando não mudou desde o último comando. Isso produz G-code mais
compacto e reduz a quantidade de dados enviados ao controlador.

### Comando Separado de Ativação do Laser para Focagem

Alguns dialetos suportam a configuração de um comando separado para ligar o
laser em baixa potência, o que é útil para o modo de focagem. Isso permite usar
um comando diferente para o comportamento visual de "ponteiro laser" daquele
usado durante o corte ou gravação real. Verifique a página de configurações do
seu dialeto para esta opção.

---

## Páginas Relacionadas

- [Exportando G-code](../files/exporting) - Configurações de exportação
- [Compatibilidade de Firmware](firmware) - Versões de firmware
- [Configurações de Dispositivo](../machine/device) - Configuração GRBL
- [Macros & Hooks](../machine/hooks-macros) - Injeção de G-code personalizado
