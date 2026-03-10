# Configurações Avançadas

A página Avançado nas Configurações da Máquina contém opções de configuração adicionais para casos de uso especializados.

![Configurações Avançadas](/screenshots/machine-advanced.png)

## Comportamento de Conexão

Configurações que controlam como o Rayforge interage com sua máquina durante a conexão.

### Origem ao Conectar

Quando habilitado, o Rayforge envia automaticamente um comando de homing ($H) ao conectar à máquina.

- **Habilite se**: Sua máquina tem chaves de limite confiáveis
- **Desabilite se**: Sua máquina não tem chaves de limite ou homing não é confiável

### Limpar Alarmes ao Conectar

Quando habilitado, o Rayforge limpa automaticamente qualquer estado de alarme ao conectar.

- **Habilite se**: Sua máquina frequentemente inicia em estado de alarme
- **Desabilite se**: Você quer investigar alarmes manualmente antes de limpar

### Permitir Homing de Eixo Único

Quando habilitado, você pode fazer homing de eixos individuais independentemente (X, Y ou Z) em vez de requerer todos os eixos juntos. Isso é útil para máquinas onde um eixo já pode estar posicionado corretamente.

## Configurações de Arcos

Configurações para controlar como caminhos curvos são convertidos em movimentos de G-code.

### Suportar Arcos

Quando habilitado, o Rayforge gera comandos de arco (G2/G3) para caminhos curvos em vez de dividi-los em muitos movimentos lineares pequenos. Isso produz G-code mais compacto e movimento mais suave na maioria dos controladores.

Quando desabilitado, todas as curvas são convertidas em segmentos lineares (comandos G1), o que fornece compatibilidade máxima com controladores que não suportam arcos.

### Tolerância de Arco

Esta configuração controla o desvio máximo permitido ao ajustar arcos a caminhos curvos, especificado em milímetros. Um valor menor produz arcos mais precisos mas pode requerer mais comandos de arco. Um valor maior permite mais desvio mas gera menos comandos.

Valores típicos variam de 0.01mm para trabalho de precisão a 0.1mm para processamento mais rápido.

## Veja Também

- [Configurações de Hardware](hardware) - Configuração de origem dos eixos e inversão
- [Configurações do Dispositivo](device) - Configurações específicas de GRBL
