# Suporte a Dialeto G-code

O Rayforge suporta múltiplos dialetos G-code para trabalhar com diferentes firmware de controlador.

## Dialeto Suportados

O Rayforge atualmente suporta estes dialetos G-code:

| Dialeto                                     | Firmware     | Uso Comum                                |
| ------------------------------------------- | ------------ | ---------------------------------------- |
| **GRBL (universal)**                        | GRBL 1.1+    | Lasers de diodo, CNC hobby               |
| **GRBL (sem eixo Z)**                       | GRBL 1.1+    | Cortadores a laser 2D sem Z              |
| **GRBL Dinâmico (Sensível à Profundidade)** | GRBL 1.1+    | Gravação a laser sensível à profundidade |
| **GRBL Dinâmico (sem eixo Z)**              | GRBL 1.1+    | Gravação a laser sensível à profundidade |
| **Mach4 (M67 Analog)**                      | Mach4        | Gravação raster de alta velocidade       |
| **Smoothieware**                            | Smoothieware | Cortadores a laser, CNC                  |
| **Marlin**                                  | Marlin 2.0+  | Impressoras 3D com laser                 |

:::note Dialeto Recomendados
:::

**GRBL (universal)** é o dialeto mais testado e recomendado para aplicações de laser padrão.

**GRBL Dinâmico (Sensível à Profundidade)** é recomendado para gravação a laser sensível à profundidade onde a potência varia durante os cortes (ex., gravação de profundidade variável).

---

## Mach4 (M67 Analog)

O dialeto **Mach4 (M67 Analog)** é projetado para gravação raster de alta velocidade com controladores Mach4. Ele usa o comando M67 com saída analógica para controle preciso de potência do laser.

### Recursos Principais

- **Saída Analógica M67**: Usa `M67 E0 Q<0-255>` para potência do laser em vez de comandos S inline
- **Pressão de Buffer Reduzida**: Ao separar comandos de potência de comandos de movimento, o buffer do controlador é menos estressado durante operações de alta velocidade
- **Raster de Alta Velocidade**: Otimizado para operações de gravação raster rápidas

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
