import React, { useState, useEffect } from 'react';
import Translate, { translate } from '@docusaurus/Translate';
import CodeBlock from '@theme/CodeBlock';
import Admonition from '@theme/Admonition';
import './InstallGuide.css';

const VERSION = typeof RAYFORGE_VERSION !== 'undefined' ? RAYFORGE_VERSION : '0.0.0';

function detectOs() {
  if (typeof window === 'undefined') {
    return 'linux';
  }

  const userAgent = window.navigator.userAgent.toLowerCase();

  if (userAgent.includes('win')) {
    return 'windows';
  }
  if (
    userAgent.includes('mac') ||
    userAgent.includes('iphone') ||
    userAgent.includes('ipad')
  ) {
    return 'macos';
  }
  if (userAgent.includes('linux')) {
    return 'linux';
  }

  return 'linux';
}

function getInitialStateFromHash() {
  if (typeof window === 'undefined') {
    return { os: null, method: null };
  }

  const hash = window.location.hash.slice(1);
  const parts = hash.split('-');
  
  if (parts.length >= 1 && ['linux', 'windows', 'macos'].includes(parts[0])) {
    const os = parts[0];
    const method = parts.length >= 2 ? parts[1] : null;
    return { os, method };
  }
  
  return { os: null, method: null };
}

const osOptions = [
  {
    id: 'linux',
    label: 'Linux',
    icon: '🐧',
  },
  {
    id: 'windows',
    label: 'Windows',
    icon: '🪟',
  },
  {
    id: 'macos',
    label: 'macOS',
    icon: '🍎',
  },
];

const linuxMethods = [
  { id: 'snap', label: translate({ id: 'install.snap.recommended', message: 'Snap (Recommended)' }) },
  { id: 'ppa', label: 'Ubuntu 24.04 (PPA)' },
  { id: 'flatpak', label: 'Flathub' },
  { id: 'pixi', label: translate({ id: 'install.pixi.developers', message: 'Pixi (Developers)' }) },
  { id: 'source', label: translate({ id: 'install.fromSource', message: 'From Source' }) },
];

const windowsMethods = [
  { id: 'installer', label: translate({ id: 'install.installer.recommended', message: 'Installer (Recommended)' }) },
  { id: 'developer', label: translate({ id: 'install.msys2.developers', message: 'MSYS2 (Developers)' }) },
];

const macosMethods = [
  { id: 'universal', label: translate({ id: 'install.macOS.universal', message: 'Universal (Recommended)' }) },
  { id: 'arm', label: translate({ id: 'install.macOS.arm', message: 'Apple Silicon (M1/M2/M3)' }) },
  { id: 'intel', label: translate({ id: 'install.macOS.intel', message: 'Intel Mac' }) },
];

function OsSelector({ selectedOs, onSelectOs }) {
  return (
    <div className="install-os-selector">
      <p><strong><Translate id="install.selectOs">Select your operating system:</Translate></strong></p>
      <div className="install-os-buttons">
        {osOptions.map((os) => (
          <button
            key={os.id}
            className={`install-os-btn ${
              selectedOs === os.id ? 'install-os-btn--active' : ''
            }`}
            onClick={() => onSelectOs(os.id)}
          >
            <span className="install-os-icon">{os.icon}</span>
            <span>{os.label}</span>
          </button>
        ))}
      </div>
    </div>
  );
}

function LinuxInstall({ method, onMethodChange }) {
  return (
    <>
      <div className="install-method-selector">
        <p><strong><Translate id="install.chooseMethod">Choose installation method:</Translate></strong></p>
        <div className="install-method-tabs">
          {linuxMethods.map((m) => (
            <button
              key={m.id}
              className={`install-method-btn ${
                method === m.id ? 'install-method-btn--active' : ''
              }`}
              onClick={() => onMethodChange(m.id)}
            >
              {m.label}
            </button>
          ))}
        </div>
      </div>

      {method === 'ppa' && <LinuxPpaInstall />}
      {method === 'flatpak' && <LinuxFlatpakInstall />}
      {method === 'snap' && <LinuxSnapInstall />}
      {method === 'pixi' && <LinuxPixiInstall />}
      {method === 'source' && <LinuxSourceInstall />}
    </>
  );
}

function LinuxPpaInstall() {
  return (
    <div className="install-section">
      <h4>Ubuntu 24.04 LTS</h4>
      <p>
        <Translate id="install.ppa.description">
          For Ubuntu 24.04 LTS, you can use our official PPA. This method
          provides automatic updates through your system's package manager.
        </Translate>
      </p>

      <div className="install-step">
        <div className="install-step-number">1</div>
        <div className="install-step-content">
          <h5><Translate id="install.addPpa">Add the Rayforge PPA</Translate></h5>
          <CodeBlock language="bash">
            {`sudo add-apt-repository ppa:knipknap/rayforge
sudo apt update`}
          </CodeBlock>
        </div>
      </div>

      <div className="install-step">
        <div className="install-step-number">2</div>
        <div className="install-step-content">
          <h5><Translate id="install.installRayforge">Install Rayforge</Translate></h5>
          <CodeBlock language="bash">sudo apt install rayforge</CodeBlock>
        </div>
      </div>

      <div className="install-step">
        <div className="install-step-number">3</div>
        <div className="install-step-content">
          <h5><Translate id="install.launchRayforge">Launch Rayforge</Translate></h5>
          <p>
            <Translate id="install.launchFromMenu">
              Launch Rayforge from your application menu or by running:
            </Translate>
          </p>
          <CodeBlock language="bash">rayforge</CodeBlock>
        </div>
      </div>
    </div>
  );
}

function LinuxFlatpakInstall() {
  return (
    <div className="install-section">
      <h4><Translate id="install.flathub.title">Flathub Package</Translate></h4>
      <p>
        <Translate id="install.flathub.description">
          The Flathub package is the easiest way to get started with Rayforge
          on Linux.
        </Translate>
      </p>

      <div className="install-step">
        <div className="install-step-number">1</div>
        <div className="install-step-content">
          <h5><Translate id="install.installFromFlathub">Install from Flathub</Translate></h5>
          <p>
            <a
              href="https://flathub.org/apps/org.rayforge.rayforge"
              target="_blank"
              rel="noopener noreferrer"
            >
              <img
                alt={translate({ id: 'install.getFromFlathub', message: 'Get it from Flathub' })}
                src="/images/flathub-badge.svg"
                height="55"
              />
            </a>
          </p>
          <p><Translate id="install.orCommandLine">Or install via command line:</Translate></p>
          <CodeBlock language="bash">
            flatpak install flathub org.rayforge.rayforge
          </CodeBlock>
        </div>
      </div>

      <div className="install-step">
        <div className="install-step-number">2</div>
        <div className="install-step-content">
          <h5><Translate id="install.launchRayforge">Launch Rayforge</Translate></h5>
          <CodeBlock language="bash">
            flatpak run org.rayforge.rayforge
          </CodeBlock>
        </div>
      </div>
    </div>
  );
}

function LinuxSnapInstall() {
  return (
    <div className="install-section">
      <h4><Translate id="install.snap.title">Snap Package</Translate></h4>
      <p>
        <Translate id="install.snap.description">
          The Snap package works on most Linux distributions and includes all
          dependencies in a sandboxed environment.
        </Translate>
      </p>

      <div className="install-step">
        <div className="install-step-number">1</div>
        <div className="install-step-content">
          <h5><Translate id="install.installRayforge">Install Rayforge</Translate></h5>
          <CodeBlock language="bash">sudo snap install rayforge</CodeBlock>
        </div>
      </div>

      <div className="install-step">
        <div className="install-step-number">2</div>
        <div className="install-step-content">
          <h5><Translate id="install.grantPermissions">Grant Permissions (Important!)</Translate></h5>
          <Admonition type="warning" title={translate({ id: 'install.permissionsRequired', message: 'Permissions Required' })}>
            <Translate id="install.snap.permissionNote">
              The Snap version runs in a sandbox and requires manual
              permission grants for hardware access.
            </Translate>
          </Admonition>

          <p>
            <strong><Translate id="install.usbSerialAccess">For USB Serial Port Access:</Translate></strong>
          </p>
          <CodeBlock language="bash">
            {`# Enable experimental hotplug support (one-time setup)
sudo snap set system experimental.hotplug=true

# Connect your laser via USB, then grant access
sudo snap connect rayforge:serial-port`}
          </CodeBlock>

          <p>
            <strong><Translate id="install.cameraAccess">For Camera Access:</Translate></strong>
          </p>
          <CodeBlock language="bash">
            sudo snap connect rayforge:camera
          </CodeBlock>
        </div>
      </div>

      <div className="install-step">
        <div className="install-step-number">3</div>
        <div className="install-step-content">
          <h5><Translate id="install.verifyPermissions">Verify Permissions</Translate></h5>
          <CodeBlock language="bash">
            snap connections rayforge
          </CodeBlock>
          <p>
            <Translate id="install.serialPortConnected" values={{ code: 'serial-port' }}>
              {'Look for {code} in the list. If it shows "connected", you\'re ready to go.'}
            </Translate>
          </p>
        </div>
      </div>
    </div>
  );
}

function LinuxPixiInstall() {
  return (
    <div className="install-section">
      <h4><Translate id="install.pixi.title">Pixi (Developer Installation)</Translate></h4>
      <p>
        <a
          href="https://pixi.sh"
          target="_blank"
          rel="noopener noreferrer"
        >
          Pixi
        </a>{' '}
        <Translate id="install.pixi.description">
          is a fast package manager for Python projects. This method is
          recommended for developers who want to contribute to Rayforge or
          run the latest development version.
        </Translate>
      </p>

      <Admonition type="info" title={translate({ id: 'install.linuxOnly', message: 'Linux Only' })}>
        <Translate id="install.pixi.linuxOnly">Pixi installation is currently only available on Linux.</Translate>
      </Admonition>

      <div className="install-step">
        <div className="install-step-number">1</div>
        <div className="install-step-content">
          <h5><Translate id="install.installPixi">Install Pixi</Translate></h5>
          <p>
            <Translate id="install.pixi.installInstruction">Install Pixi using the official installer:</Translate>
          </p>
          <CodeBlock language="bash">
            curl -fsSL https://pixi.sh/install.sh | bash
          </CodeBlock>
          <p>
            <Translate id="install.restartShell">After installation, restart your shell or run:</Translate>
          </p>
          <CodeBlock language="bash">source ~/.bashrc</CodeBlock>
        </div>
      </div>

      <div className="install-step">
        <div className="install-step-number">2</div>
        <div className="install-step-content">
          <h5><Translate id="install.cloneRepo">Clone the Repository</Translate></h5>
          <CodeBlock language="bash">
            {`git clone https://github.com/barebaric/rayforge.git
cd rayforge`}
          </CodeBlock>
        </div>
      </div>

      <div className="install-step">
        <div className="install-step-number">3</div>
        <div className="install-step-content">
          <h5><Translate id="install.installSystemDependency">Install System Dependency</Translate></h5>
          <p><Translate id="install.installGtkAdwaita">Install the required Gtk/Adwaita package:</Translate></p>
          <CodeBlock language="bash">sudo apt install gir1.2-adw-1</CodeBlock>
        </div>
      </div>

      <div className="install-step">
        <div className="install-step-number">4</div>
        <div className="install-step-content">
          <h5><Translate id="install.installDependencies">Install Dependencies</Translate></h5>
          <p>
            <Translate id="install.pixi.autoVenv">
              Pixi will automatically create a virtual environment and
              install all dependencies:
            </Translate>
          </p>
          <CodeBlock language="bash">pixi install</CodeBlock>
        </div>
      </div>

      <div className="install-step">
        <div className="install-step-number">5</div>
        <div className="install-step-content">
          <h5><Translate id="install.addDialoutGroup">Add User to dialout Group</Translate></h5>
          <p><Translate id="install.requiredSerialAccess">Required for serial port access:</Translate></p>
          <CodeBlock language="bash">
            sudo usermod -a -G dialout $USER
          </CodeBlock>
          <p>
            <strong><Translate id="install.important">Important:</Translate></strong>{' '}
            <Translate id="install.logoutLogin">Log out and log back in for this change to take effect.</Translate>
          </p>
        </div>
      </div>

      <div className="install-step">
        <div className="install-step-number">6</div>
        <div className="install-step-content">
          <h5><Translate id="install.runRayforge">Run Rayforge</Translate></h5>
          <CodeBlock language="bash">pixi run rayforge</CodeBlock>
        </div>
      </div>

      <div className="install-troubleshoot">
        <h5><Translate id="install.usefulPixiCommands">Useful Pixi Commands</Translate></h5>
        <details>
          <summary><Translate id="install.availableDevCommands">Available development commands</Translate></summary>
          <div className="install-troubleshoot-content">
            <ul>
              <li>
                <code>pixi run test</code> - <Translate id="install.runTestSuite">Run the test suite</Translate>
              </li>
              <li>
                <code>pixi run uitest</code> - <Translate id="install.runUiTests">Run UI tests</Translate>
              </li>
              <li>
                <code>pixi run lint</code> - <Translate id="install.runLinting">Run linting and static analysis</Translate>
              </li>
              <li>
                <code>pixi run format</code> - <Translate id="install.formatCode">Format code with ruff</Translate>
              </li>
            </ul>
          </div>
        </details>
      </div>
    </div>
  );
}

function LinuxSourceInstall() {
  return (
    <div className="install-section">
      <h4><Translate id="install.source.title">Install from Source</Translate></h4>
      <p>
        <Translate id="install.source.description">
          For developers and advanced users who want to run the latest
          development version or contribute to the project.
        </Translate>
      </p>

      <div className="install-step">
        <div className="install-step-number">1</div>
        <div className="install-step-content">
          <h5><Translate id="install.installSystemDeps">Install System Dependencies</Translate></h5>
          <p><Translate id="install.onDebianUbuntu">On Debian/Ubuntu:</Translate></p>
          <CodeBlock language="bash">
            {`sudo apt update
sudo apt install python3-pip python3-gi gir1.2-gtk-3.0 gir1.2-adw-1 \\
  gir1.2-gdkpixbuf-2.0 libgirepository-1.0-dev libgirepository-2.0-0 \\
  libvips42t64 libadwaita-1-0 libopencv-dev`}
          </CodeBlock>
        </div>
      </div>

      <div className="install-step">
        <div className="install-step-number">2</div>
        <div className="install-step-content">
          <h5><Translate id="install.installFromPypi">Install Rayforge from PyPI</Translate></h5>
          <CodeBlock language="bash">pip3 install rayforge</CodeBlock>
          <Admonition type="note">
            <Translate id="install.pypiNote">
              Package names may differ on other distributions. Refer to your
              distribution's documentation.
            </Translate>
          </Admonition>
        </div>
      </div>

      <div className="install-step">
        <div className="install-step-number">3</div>
        <div className="install-step-content">
          <h5><Translate id="install.addDialoutGroup">Add User to dialout Group</Translate></h5>
          <p><Translate id="install.requiredSerialAccess">Required for serial port access:</Translate></p>
          <CodeBlock language="bash">
            sudo usermod -a -G dialout $USER
          </CodeBlock>
          <p>
            <strong><Translate id="install.important">Important:</Translate></strong>{' '}
            <Translate id="install.logoutLogin">Log out and log back in for this change to take effect.</Translate>
          </p>
        </div>
      </div>
    </div>
  );
}

function WindowsInstall({ version, method, onMethodChange }) {
  return (
    <>
      <div className="install-method-selector">
        <p><strong><Translate id="install.chooseMethod">Choose installation method:</Translate></strong></p>
        <div className="install-method-tabs">
          {windowsMethods.map((m) => (
            <button
              key={m.id}
              className={`install-method-btn ${
                method === m.id ? 'install-method-btn--active' : ''
              }`}
              onClick={() => onMethodChange(m.id)}
            >
              {m.label}
            </button>
          ))}
        </div>
      </div>

      {method === 'installer' && <WindowsInstallerInstall version={version} />}
      {method === 'developer' && <WindowsDeveloperInstall />}
    </>
  );
}

function WindowsInstallerInstall({ version }) {
  const downloadUrl = `https://github.com/barebaric/rayforge/releases/download/${version}/rayforge-v${version}-installer.exe`;
  return (
    <div className="install-section">
      <h4><Translate id="install.windows.title">Windows Installation</Translate></h4>

      <div className="install-step">
        <div className="install-step-number">1</div>
        <div className="install-step-content">
          <h5><Translate id="install.downloadInstaller">Download the Installer</Translate></h5>
          <p>
            <a
              href={downloadUrl}
              target="_blank"
              rel="noopener noreferrer"
            >
              <strong>{translate({ id: 'install.downloadRayforge', message: 'Download Rayforge' }, { version })}</strong>
            </a>
          </p>
        </div>
      </div>

      <div className="install-step">
        <div className="install-step-number">2</div>
        <div className="install-step-content">
          <h5><Translate id="install.runInstaller">Run the Installer</Translate></h5>
          <p>
            <Translate id="install.runInstallerInstruction">
              Run the downloaded installer and follow the on-screen instructions.
            </Translate>
          </p>
          <Admonition type="tip">
            <Translate id="install.runAsAdmin">
              If you encounter permission errors, try running the installer
              as Administrator (right-click - Run as Administrator).
            </Translate>
          </Admonition>
        </div>
      </div>

      <div className="install-step">
        <div className="install-step-number">3</div>
        <div className="install-step-content">
          <h5><Translate id="install.launchRayforge">Launch Rayforge</Translate></h5>
          <p>
            <Translate id="install.windowsLaunch">
              Launch Rayforge from the Start Menu or Desktop shortcut.
            </Translate>
          </p>
        </div>
      </div>
    </div>
  );
}

function WindowsDeveloperInstall() {
  return (
    <div className="install-section">
      <h4><Translate id="install.windows.dev.title">Windows Developer Setup (MSYS2)</Translate></h4>
      <p>
        <Translate id="install.windows.dev.description">
          For developers who want to contribute to Rayforge or run the latest
          development version on Windows. This method uses MSYS2 to provide a
          Unix-like development environment.
        </Translate>
      </p>

      <div className="install-step">
        <div className="install-step-number">1</div>
        <div className="install-step-content">
          <h5><Translate id="install.installMsys2">Install MSYS2</Translate></h5>
          <p>
            <Translate id="install.msys2.download">
              Download and install MSYS2 from the official website:
            </Translate>
          </p>
          <p>
            <a
              href="https://www.msys2.org/"
              target="_blank"
              rel="noopener noreferrer"
            >
              <strong>msys2.org</strong>
            </a>
          </p>
          <Admonition type="tip">
            <Translate id="install.msys2.defaultPath">
              Use the default installation path (C:\msys64) for best compatibility.
            </Translate>
          </Admonition>
        </div>
      </div>

      <div className="install-step">
        <div className="install-step-number">2</div>
        <div className="install-step-content">
          <h5><Translate id="install.cloneRepo">Clone the Repository</Translate></h5>
          <CodeBlock language="bash">
            {`git clone https://github.com/barebaric/rayforge.git
cd rayforge`}
          </CodeBlock>
        </div>
      </div>

      <div className="install-step">
        <div className="install-step-number">3</div>
        <div className="install-step-content">
          <h5><Translate id="install.runSetup">Run the Setup Script</Translate></h5>
          <p>
            <Translate id="install.msys2.shell">
              Open the MSYS2 MINGW64 shell and run the setup script:
            </Translate>
          </p>
          <CodeBlock language="bash">
            bash scripts/win/win_setup.sh
          </CodeBlock>
          <p>
            <Translate id="install.msys2.setupNote">
              This script installs all required system dependencies and Python packages.
              It may take several minutes to complete.
            </Translate>
          </p>
        </div>
      </div>

      <div className="install-step">
        <div className="install-step-number">4</div>
        <div className="install-step-content">
          <h5><Translate id="install.installDevTools">Install Development Tools (Optional)</Translate></h5>
          <p>
            <Translate id="install.msys2.devTools">
              For linting, formatting, and pre-commit hooks:
            </Translate>
          </p>
          <CodeBlock language="bash">
            bash scripts/win/win_setup_dev.sh
          </CodeBlock>
        </div>
      </div>

      <div className="install-step">
        <div className="install-step-number">5</div>
        <div className="install-step-content">
          <h5><Translate id="install.runRayforge">Run Rayforge</Translate></h5>
          <CodeBlock language="bash">
            bash scripts/win/win_run.sh
          </CodeBlock>
        </div>
      </div>

      <div className="install-troubleshoot">
        <h5><Translate id="install.usefulWinCommands">Useful Windows Development Commands</Translate></h5>
        <details>
          <summary><Translate id="install.availableDevCommands">Available development commands</Translate></summary>
          <div className="install-troubleshoot-content">
            <ul>
              <li>
                <code>bash scripts/win/win_run.sh</code> - <Translate id="install.runRayforge">Run Rayforge</Translate>
              </li>
              <li>
                <code>bash scripts/win/win_test.sh</code> - <Translate id="install.runTestSuite">Run the test suite</Translate>
              </li>
              <li>
                <code>bash scripts/win/win_lint.sh</code> - <Translate id="install.runLinting">Run linting and static analysis</Translate>
              </li>
              <li>
                <code>bash scripts/win/win_format.sh</code> - <Translate id="install.formatCode">Format code with ruff</Translate>
              </li>
              <li>
                <code>bash scripts/win/win_build.sh</code> - <Translate id="install.buildInstaller">Build the Windows installer</Translate>
              </li>
            </ul>
          </div>
        </details>
      </div>

      <Admonition type="note" title={translate({ id: 'install.gitCommands', message: 'Git Commands' })}>
        <Translate id="install.msys2.gitNote">
          When using pre-commit hooks, you must run git commands from within the
          MSYS2 MINGW64 shell, not from PowerShell or Command Prompt.
        </Translate>
      </Admonition>
    </div>
  );
}

function MacosInstall({ version, method, onMethodChange }) {
  const downloadUrl = `https://github.com/barebaric/rayforge/releases/download/${version}/rayforge-v${version}-macos`;
  return (
    <div className="install-section">
      <h4><Translate id="install.macOS.title">macOS Installation</Translate></h4>

      <div className="install-method-selector">
        <p><strong><Translate id="install.macOS.chooseBuild">Choose build type:</Translate></strong></p>
        <div className="install-method-tabs">
          {macosMethods.map((m) => (
            <button
              key={m.id}
              className={`install-method-btn ${
                method === m.id ? 'install-method-btn--active' : ''
              }`}
              onClick={() => onMethodChange(m.id)}
            >
              {m.label}
            </button>
          ))}
        </div>
      </div>

      <div className="install-step">
        <div className="install-step-number">1</div>
        <div className="install-step-content">
          <h5><Translate id="install.downloadInstaller">Download the Installer</Translate></h5>
          {method === 'universal' && (
            <>
              <p>
                <a
                  href={`${downloadUrl}-universal.dmg`}
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  <strong>{translate({ id: 'install.downloadRayforge', message: 'Download Rayforge' }, { version })}</strong>
                </a>
              </p>
              <p>
                <Translate id="install.macOS.universalInstructions">
                  Open the DMG and drag Rayforge.app to your Applications folder.
                </Translate>
              </p>
            </>
          )}
          {method === 'arm' && (
            <>
              <p>
                <a
                  href={`${downloadUrl}-arm-app.zip`}
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  <strong>{translate({ id: 'install.downloadRayforge', message: 'Download Rayforge' }, { version })}</strong>
                </a>
              </p>
              <p>
                <Translate id="install.macOS.zipInstructions">
                  Extract the ZIP and move Rayforge.app to Applications.
                </Translate>
              </p>
            </>
          )}
          {method === 'intel' && (
            <>
              <p>
                <a
                  href={`${downloadUrl}-intel-app.zip`}
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  <strong>{translate({ id: 'install.downloadRayforge', message: 'Download Rayforge' }, { version })}</strong>
                </a>
              </p>
              <p>
                <Translate id="install.macOS.zipInstructions">
                  Extract the ZIP and move Rayforge.app to Applications.
                </Translate>
              </p>
            </>
          )}
        </div>
      </div>

      <div className="install-step">
        <div className="install-step-number">2</div>
        <div className="install-step-content">
          <h5><Translate id="install.macOS.openRayforge">Open Rayforge</Translate></h5>
          <p>
            <Translate id="install.macOS.openRayforgeInstructions">
              Start Rayforge.app from your Applications folder.
            </Translate>
          </p>
        </div>
      </div>

      <div className="install-step">
        <div className="install-step-number">3</div>
        <div className="install-step-content">
          <h5><Translate id="install.macOS.gatekeeper">Launch Once from Terminal (if Gatekeeper blocks it)</Translate></h5>
          <p>
            <Translate id="install.macOS.gatekeeperInstructions">
              If macOS blocks the first launch, remove quarantine attributes and start Rayforge:
            </Translate>
          </p>
          <CodeBlock language="bash">
            {`xattr -dr com.apple.quarantine /Applications/Rayforge.app
/Applications/Rayforge.app/Contents/MacOS/Rayforge --version`}
          </CodeBlock>
        </div>
      </div>

      <details className="install-details">
        <summary><Translate id="install.macOS.buildFromSource">Build from source (advanced)</Translate></summary>
        <div className="install-details-content">
          <p>
            <Translate id="install.macOS.buildFromSourceInstructions">
              If you prefer a local source build, use the macOS setup/build scripts from the repository:
            </Translate>
          </p>
          <CodeBlock language="bash">
            {`git clone https://github.com/barebaric/rayforge.git
cd rayforge
bash scripts/mac/mac_setup.sh --install
source .mac_env
printf "5\\n" | bash scripts/mac/mac_build.sh
dist/Rayforge.app/Contents/MacOS/Rayforge --version`}
          </CodeBlock>
        </div>
      </details>
    </div>
  );
}

function VerifyInstall({ os }) {
  return (
    <div className="install-section">
      <h4><Translate id="install.verify.title">Verify Installation</Translate></h4>

      <div className="install-step">
        <div className="install-step-number">✓</div>
        <div className="install-step-content">
          <h5><Translate id="install.launchRayforge">Launch Rayforge</Translate></h5>
          <p>
            <Translate id="install.verify.launch">
              Launch Rayforge from your application menu or terminal. You
              should see the main window with the canvas and toolbar.
            </Translate>
          </p>
        </div>
      </div>

      <div className="install-step">
        <div className="install-step-number">✓</div>
        <div className="install-step-content">
          <h5><Translate id="install.checkVersion">Check Version</Translate></h5>
          <p>
            <Translate id="install.checkVersionInstruction">
              Check Help - About to confirm the installed version.
            </Translate>
          </p>
        </div>
      </div>

      <div className="install-step">
        <div className="install-step-number">✓</div>
        <div className="install-step-content">
          <h5><Translate id="install.connectMachine">Connect Your Machine</Translate></h5>
          <p>
            <Translate id="install.verifySerialPort">
              Connect your laser controller via USB and verify that the
              serial port appears in Rayforge's machine settings.
            </Translate>
          </p>
          {os === 'linux' && (
            <Admonition type="tip">
              <Translate id="install.linuxSerialTip">
                On Linux, laser controllers typically appear as /dev/ttyUSB0 or /dev/ttyACM0.
              </Translate>
            </Admonition>
          )}
          {os === 'windows' && (
            <Admonition type="tip">
              <Translate id="install.windowsComTip">
                On Windows, check Device Manager under Ports (COM and LPT) to find your COM port number.
              </Translate>
            </Admonition>
          )}
        </div>
      </div>
    </div>
  );
}

function Troubleshooting({ os, linuxMethod, windowsMethod }) {
  return (
    <div className="install-section">
      <h4><Translate id="install.troubleshooting.title">Troubleshooting</Translate></h4>
      {os === 'linux' && linuxMethod === 'snap' && (
        <div className="install-troubleshoot">
          <details>
            <summary><Translate id="install.permissionIssues">Permission issues?</Translate></summary>
            <div className="install-troubleshoot-content">
              <p>
                <Translate id="install.seeSnapGuide">
                  If your device is not detected, see the Snap Permissions Guide.
                </Translate>
                {' '}
                <a href="../troubleshooting/snap-permissions">
                  <Translate id="install.snapPermissionsGuide">Snap Permissions Guide</Translate>
                </a>
              </p>
            </div>
          </details>
          <details>
            <summary><Translate id="install.serialNotWorking">Serial port still not working?</Translate></summary>
            <div className="install-troubleshoot-content">
              <ol>
                <li>
                  <strong><Translate id="install.replugUsb">Replug the USB device:</Translate></strong>{' '}
                  <Translate id="install.replugInstruction">
                    Unplug your laser controller, wait 5 seconds, plug it back in.
                  </Translate>
                </li>
                <li>
                  <strong><Translate id="install.restartRayforge">Restart Rayforge:</Translate></strong>{' '}
                  <Translate id="install.closeRelaunch">Close completely and relaunch.</Translate>
                </li>
                <li>
                  <strong><Translate id="install.checkDeviceExists">Check the device exists:</Translate></strong>
                  <CodeBlock language="bash">
                    ls -l /dev/ttyUSB* /dev/ttyACM*
                  </CodeBlock>
                </li>
                <li>
                  <strong><Translate id="install.reinstallIfNeeded">Reinstall if needed:</Translate></strong>
                  <CodeBlock language="bash">
                    {`sudo snap remove rayforge
sudo snap install rayforge
sudo snap connect rayforge:serial-port`}
                  </CodeBlock>
                </li>
              </ol>
            </div>
          </details>
        </div>
      )}
      {os === 'linux' && linuxMethod !== 'snap' && (
        <div className="install-troubleshoot">
          <details>
            <summary><Translate id="install.permissionDenied">Permission denied errors?</Translate></summary>
            <div className="install-troubleshoot-content">
              <p>
                <Translate id="install.permissionDeniedInstruction" values={{ code: 'dialout' }}>
                  {'If you get permission denied errors when accessing the serial port, add your user to the {code} group:'}
                </Translate>
              </p>
              <CodeBlock language="bash">
                sudo usermod -a -G dialout $USER
              </CodeBlock>
              <p><Translate id="install.logoutBack">Then log out and back in for changes to take effect.</Translate></p>
            </div>
          </details>
        </div>
      )}
      {os === 'windows' && (
        <div className="install-troubleshoot">
          <details>
            <summary><Translate id="install.installerCrashing">Installer crashing or not installing?</Translate></summary>
            <div className="install-troubleshoot-content">
              <p>
                <Translate id="install.missingVcRedist">
                  If the installer crashes or fails to install, you may be
                  missing the Microsoft Visual C++ Redistributable. Download
                  and install it from Microsoft:
                </Translate>
              </p>
              <p>
                <a
                  href="https://aka.ms/vc14/vc_redist.x64.exe"
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  <strong><Translate id="install.downloadVc14">Download C++ Redistributable v14</Translate></strong>
                </a>
              </p>
              <p>
                <Translate id="install.afterVcInstall">
                  After installing the redistributable, run the Rayforge installer again.
                </Translate>
              </p>
            </div>
          </details>
          <details>
            <summary><Translate id="install.driverIssues">Driver issues?</Translate></summary>
            <div className="install-troubleshoot-content">
              <p>
                <Translate id="install.checkDeviceManager">
                  If your laser controller is not detected, check Device
                  Manager under Ports (COM and LPT) to confirm the COM port number.
                  You may need to install CH340 or CP2102 drivers for your USB-to-serial adapter.
                </Translate>
              </p>
            </div>
          </details>
          <details>
            <summary><Translate id="install.antivirusBlocking">Antivirus blocking connection?</Translate></summary>
            <div className="install-troubleshoot-content">
              <p>
                <Translate id="install.antivirusNote">
                  Some antivirus software may block Rayforge from accessing USB
                  devices. Try adding an exception if you experience connection issues.
                </Translate>
              </p>
            </div>
          </details>
        </div>
      )}
      {os === 'macos' && (
        <div className="install-troubleshoot">
          <details>
            <summary><Translate id="install.usbNotDetected">USB device not detected?</Translate></summary>
            <div className="install-troubleshoot-content">
              <p>
                <Translate id="install.approveUsb">
                  If your device is not detected, you may need to approve the
                  USB device in System Settings - Privacy and Security.
                </Translate>
              </p>
            </div>
          </details>
          <details>
            <summary><Translate id="install.driverIssues">Driver issues?</Translate></summary>
            <div className="install-troubleshoot-content">
              <p>
                <Translate id="install.driverIssuesMac">
                  Some USB-to-serial adapters require additional drivers. Check
                  if your adapter uses CH340 or CP2102 chips and install the
                  appropriate driver.
                </Translate>
              </p>
            </div>
          </details>
        </div>
      )}
    </div>
  );
}

function NeedHelp() {
  return (
    <div className="install-section">
      <h4><Translate id="install.needHelp">Need Help?</Translate></h4>
      <p>
        <Translate id="install.needHelpDescription">
          For additional help, report an issue on GitHub or join our Discord.
        </Translate>
        {' '}
        <a href="https://github.com/barebaric/rayforge/issues">GitHub</a>
        {' | '}
        <a href="https://discord.gg/sTHNdTtpQJ">Discord</a>
      </p>
    </div>
  );
}

export default function InstallGuide() {
  const hashState = getInitialStateFromHash();
  const [selectedOs, setSelectedOs] = useState(() => hashState.os || detectOs());
  const [linuxMethod, setLinuxMethod] = useState(() => hashState.method || 'snap');
  const [windowsMethod, setWindowsMethod] = useState(() => hashState.method || 'installer');
  const [macosMethod, setMacosMethod] = useState(() => hashState.method || 'universal');

  useEffect(() => {
    const detected = detectOs();
    const hashState = getInitialStateFromHash();
    if (hashState.os) {
      setSelectedOs(hashState.os);
      if (hashState.method) {
        if (hashState.os === 'linux') {
          setLinuxMethod(hashState.method);
        }
        if (hashState.os === 'windows') {
          setWindowsMethod(hashState.method);
        }
        if (hashState.os === 'macos') {
          setMacosMethod(hashState.method);
        }
      }
    } else {
      setSelectedOs(detected);
    }
  }, []);

  return (
    <div className="install-guide">
      <OsSelector selectedOs={selectedOs} onSelectOs={setSelectedOs} />

      <div className="install-content">
        {selectedOs === 'linux' && (
          <LinuxInstall
            method={linuxMethod}
            onMethodChange={setLinuxMethod}
          />
        )}
        {selectedOs === 'windows' && (
          <WindowsInstall
            version={VERSION}
            method={windowsMethod}
            onMethodChange={setWindowsMethod}
          />
        )}
        {selectedOs === 'macos' && (
          <MacosInstall
            version={VERSION}
            method={macosMethod}
            onMethodChange={setMacosMethod}
          />
        )}

        <VerifyInstall os={selectedOs} />
        <Troubleshooting os={selectedOs} linuxMethod={linuxMethod} windowsMethod={windowsMethod} />
        <NeedHelp />
      </div>
    </div>
  );
}
