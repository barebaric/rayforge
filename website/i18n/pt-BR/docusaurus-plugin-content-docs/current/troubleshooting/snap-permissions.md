# Permissões Snap (Linux)

Esta página explica como configurar permissões para o Rayforge quando instalado como pacote Snap no Linux.

## O Que São Permissões Snap?

Snaps são aplicações containerizadas que rodam em uma sandbox para segurança. Por padrão, elas têm acesso limitado a recursos do sistema. Para usar certos recursos (como portas seriais para controladores de laser), você deve conceder permissões explicitamente.

## Permissões Necessárias

O Rayforge precisa destas interfaces Snap conectadas para funcionalidade completa:

| Interface | Propósito | Obrigatório? |
|-----------|---------|-----------|
| `serial-port` | Acesso a dispositivos seriais USB (controladores de laser) | **Sim** (para controle de máquina) |
| `home` | Ler/escrever arquivos no seu diretório home | Conectado automaticamente |
| `removable-media` | Acessar drives externos e armazenamento USB | Opcional |
| `network` | Conectividade de rede (para atualizações, etc.) | Conectado automaticamente |

---

## Concedendo Acesso à Porta Serial

**Esta é a permissão mais importante para o Rayforge.**

### Verifique Permissões Atuais

```bash
# Veja todas as conexões para o Rayforge
snap connections rayforge
```

Procure pela interface `serial-port`. Se mostrar "desconectado" ou "-", você precisa conectá-la.

### Conecte a Interface de Porta Serial

```bash
# Conceda acesso à porta serial
sudo snap connect rayforge:serial-port
```

**Você só precisa fazer isso uma vez.** A permissão persiste entre atualizações do app e reinicializações.

### Verifique a Conexão

```bash
# Verifique se serial-port está conectado agora
snap connections rayforge | grep serial-port
```

Saída esperada:
```
serial-port     rayforge:serial-port     :serial-port     -
```

Se você vir um indicador plug/slot, a conexão está ativa.

---

## Concedendo Acesso a Mídia Removível

Se você quer importar/exportar arquivos de drives USB ou armazenamento externo:

```bash
# Conceda acesso a mídia removível
sudo snap connect rayforge:removable-media
```

Agora você pode acessar arquivos em `/media` e `/mnt`.

---

## Solução de Problemas de Permissões Snap

### Porta Serial Ainda Não Funciona

**Após conectar a interface:**

1. ** Reconecte o dispositivo USB:**
   - Desconecte seu controlador de laser
   - Espere 5 segundos
   - Conecte novamente

2. ** Reinicie o Rayforge:**
   - Feche o Rayforge completamente
   - Relance a partir do menu de aplicativos ou:
     ```bash
     snap run rayforge
     ```

3. ** Verifique se a porta aparece:**
   - Abra Rayforge  Configurações  Máquina
   - Procure portas seriais no menu suspenso
   - Deve ver `/dev/ttyUSB0`, `/dev/ttyACM0`, ou similar

4. ** Verifique se o dispositivo existe:**
   ```bash
   # Liste dispositivos seriais USB
   ls -l /dev/ttyUSB* /dev/ttyACM*
   ```

### "Permissão Negada" Apesar da Interface Conectada

Isso é raro mas pode acontecer se:

1. ** A instalação do Snap está quebrada:**
   ```bash
   # Reinstale o snap
   sudo snap refresh rayforge --devmode
   # Ou se isso falhar:
   sudo snap remove rayforge
   sudo snap install rayforge
   # Reconecte as interfaces
   sudo snap connect rayforge:serial-port
   ```

2. ** Regras udev conflitantes:**
   - Verifique `/etc/udev/rules.d/` por regras personalizadas de porta serial
   - Elas podem conflitar com o acesso a dispositivos do Snap

3. ** Negações AppArmor:**
   ```bash
   # Verifique por negações AppArmor
   sudo journalctl -xe | grep DENIED | grep rayforge
   ```

   Se você vir negações para portas seriais, pode haver um conflito de perfil AppArmor.

### Não Consegue Acessar Arquivos Fora do Diretório Home

**Por design**, Snaps não podem acessar arquivos fora do seu diretório home a menos que você conceda `removable-media`.

**Opções de workaround:**

1. ** Mova arquivos para seu diretório home:**
   ```bash
   # Copie arquivos SVG para ~/Documents
   cp /algum/outro/local/*.svg ~/Documents/
   ```

2. ** Conceda acesso removable-media:**
   ```bash
   sudo snap connect rayforge:removable-media
   ```

3. ** Use o seletor de arquivos do Snap:**
   - O seletor de arquivos embutido tem acesso mais amplo
   - Abra arquivos através de Arquivo  Abrir em vez de argumentos de linha de comando

---

## Gerenciamento Manual de Interfaces

### Liste Todas as Interfaces Disponíveis

```bash
# Veja todas as interfaces Snap no seu sistema
snap interface
```

### Desconecte uma Interface

```bash
# Desconecte serial-port (se necessário)
sudo snap disconnect rayforge:serial-port
```

### Reconecte Após Desconectar

```bash
sudo snap connect rayforge:serial-port
```

---

## Alternativa: Instale a Partir do Código Fonte

Se as permissões Snap forem muito restritivas para seu fluxo de trabalho:

**Opção 1: Compile a partir do código fonte**

```bash
# Clone o repositório
git clone https://github.com/kylemartin57/rayforge.git
cd rayforge

# Instale dependências usando pixi
pixi install

# Execute o Rayforge
pixi run rayforge
```

**Benefícios:**
- Sem restrições de permissão
- Acesso completo ao sistema
- Depuração mais fácil
- Versão de desenvolvimento mais recente

**Desvantagens:**
- Atualizações manuais (git pull)
- Mais dependências para gerenciar
- Sem atualizações automáticas

**Opção 2: Use Flatpak (se disponível)**

Flatpak tem sandboxing similar mas às vezes com modelos de permissão diferentes. Verifique se o Rayforge oferece um pacote Flatpak.

---

## Melhores Práticas de Permissão Snap

### Conecte Apenas o Que Você Precisa

Não conecte interfaces que você não usa:

-  Conecte `serial-port` se você usa um controlador de laser
-  Conecte `removable-media` se você importa de drives USB
- L Não conecte tudo "por precaução" - derrota o propósito de segurança

### Verifique a Fonte do Snap

Sempre instale da Snap Store oficial:

```bash
# Verifique o publicador
snap info rayforge
```

Procure por:
- Publicador verificado
- Fonte de repositório oficial
- Atualizações regulares

---

## Entendendo a Sandbox Snap

### O Que Snaps Podem Acessar por Padrão?

**Permitido:**
- Arquivos no seu diretório home
- Conexões de rede
- Display/áudio

**Não permitido sem permissão explícita:**
- Portas seriais (dispositivos USB)
- Mídia removível
- Arquivos de sistema
- Diretórios home de outros usuários

### Por Que Isso Importa para o Rayforge

O Rayforge precisa:

1. ** Acesso ao diretório home** (concedido automaticamente)
   - Para salvar arquivos de projeto
   - Para ler arquivos SVG/DXF importados
   - Para armazenar preferências

2. ** Acesso à porta serial** (deve ser concedido)
   - Para comunicar com controladores de laser
   - **Esta é a permissão crítica**

3. ** Mídia removível** (opcional)
   - Para importar arquivos de drives USB
   - Para exportar G-code para armazenamento externo

---

## Depurando Problemas de Snap

### Habilite Logging Verboso do Snap

```bash
# Execute o Snap com saída de depuração
snap run --shell rayforge
# Dentro do shell do snap:
export RAYFORGE_LOG_LEVEL=DEBUG
exec rayforge
```

### Verifique os Logs do Snap

```bash
# Veja logs do Rayforge
snap logs rayforge

# Siga logs em tempo real
snap logs -f rayforge
```

### Verifique o Journal do Sistema para Negações

```bash
# Procure por negações AppArmor
sudo journalctl -xe | grep DENIED | grep rayforge

# Procure por eventos de dispositivo USB
sudo journalctl -f -u snapd
# Então conecte seu controlador de laser
```

---

## Obtendo Ajuda

Se você ainda está tendo problemas relacionados ao Snap:

1. ** Verifique as permissões primeiro:**
   ```bash
   snap connections rayforge
   ```

2. ** Tente um teste de porta serial:**
   ```bash
   # Se você tem screen ou minicom instalado
   sudo snap connect rayforge:serial-port
   # Então teste no Rayforge
   ```

3. ** Reporte o problema com:**
   - Saída de `snap connections rayforge`
   - Saída de `snap version`
   - Saída de `snap info rayforge`
   - Sua versão de distribuição Ubuntu/Linux
   - Mensagens de erro exatas

4. ** Considere alternativas:**
   - Instale a partir do código fonte (veja acima)
   - Use um formato de pacote diferente (AppImage, Flatpak)

---

## Comandos de Referência Rápida

```bash
# Conceda acesso à porta serial (mais importante)
sudo snap connect rayforge:serial-port

# Conceda acesso a mídia removível
sudo snap connect rayforge:removable-media

# Verifique conexões atuais
snap connections rayforge

# Veja logs do Rayforge
snap logs rayforge

# Atualize o Rayforge
sudo snap refresh rayforge

# Remova e reinstale (último recurso)
sudo snap remove rayforge
sudo snap install rayforge
sudo snap connect rayforge:serial-port
```

---

## Páginas Relacionadas

- [Problemas de Conexão](connection) - Solução de problemas de conexão serial
- [Modo de Depuração](debug) - Habilite logging de diagnóstico
- [Instalação](../getting-started/installation) - Guia de instalação
- [Configurações Gerais](../machine/general) - Configuração de máquina
