# Suporte a Dialeto G-code

O Rayforge suporta múltiplos dialetos G-code para trabalhar com diferentes firmware de controlador.

## Dialeto Suportados

O Rayforge atualmente suporta estes dialetos G-code:

| Dialeto                        | Firmware     | Uso Comum                  | Status                          |
| ------------------------------ | ------------ | --------------------------- | ------------------------------- |
| **GRBL (universal)**           | GRBL 1.1+    | Lasers de diodo, CNC hobby     |  Primário, totalmente suportado      |
| **GRBL (sem eixo Z)**           | GRBL 1.1+    | Cortadores a laser 2D sem Z  |  Variante otimizada             |
| **GRBL Dinâmico (Sensível à Profundidade)** | GRBL 1.1+    | Gravação a laser sensível à profundidade |  Recomendado para potência dinâmica |
| **GRBL Dinâmico (sem eixo Z)**   | GRBL 1.1+    | Gravação a laser sensível à profundidade |  Variante otimizada             |
| **Smoothieware**               | Smoothieware | Cortadores a laser, CNC          |  Experimental                  |
| **Marlin**                     | Marlin 2.0+  | Impressoras 3D com laser      |  Experimental                  |

:::note Dialeto Recomendados
:::

**GRBL (universal)** é o dialeto mais testado e recomendado para aplicações de laser padrão.

    **GRBL Dinâmico (Sensível à Profundidade)** é recomendado para gravação a laser sensível à profundidade onde a potência varia durante os cortes (ex., gravação de profundidade variável).
---

## Criando um Dialeto Personalizado

Para criar um dialeto G-code personalizado baseado em um dialeto embutido:

1. Abra **Configurações de Máquina** → **Dialeto G-code**
2. Clique no ícone **Copiar** em um dialeto embutido para criar um novo dialeto personalizado
3. Edite as configurações do dialeto conforme necessário
4. Salve seu dialeto personalizado

Dialeto personalizados são armazenados no seu diretório de configuração e podem ser compartilhados.

---

## Páginas Relacionadas

- [Exportando G-code](../files/exporting) - Configurações de exportação
- [Compatibilidade de Firmware](firmware) - Versões de firmware
- [Configurações de Dispositivo](../machine/device) - Configuração GRBL
- [Macros & Hooks](../machine/hooks-macros) - Injeção de G-code personalizado
