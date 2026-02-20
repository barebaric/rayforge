import React, { useState, useEffect } from 'react';
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
  { id: 'snap', label: 'Snap (Recommended)' },
  { id: 'ppa', label: 'Ubuntu 24.04 (PPA)' },
  { id: 'flatpak', label: 'Flathub' },
  { id: 'pixi', label: 'Pixi (Developers)' },
  { id: 'source', label: 'From Source' },
];

function OsSelector({ selectedOs, onSelectOs }) {
  return (
    <div className="install-os-selector">
      <p><strong>Select your operating system:</strong></p>
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
        <p><strong>Choose installation method:</strong></p>
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
        For Ubuntu 24.04 LTS, you can use our official PPA. This method
        provides automatic updates through your system's package manager.
      </p>

      <div className="install-step">
        <div className="install-step-number">1</div>
        <div className="install-step-content">
          <h5>Add the Rayforge PPA</h5>
          <CodeBlock language="bash">
            {`sudo add-apt-repository ppa:knipknap/rayforge
sudo apt update`}
          </CodeBlock>
        </div>
      </div>

      <div className="install-step">
        <div className="install-step-number">2</div>
        <div className="install-step-content">
          <h5>Install Rayforge</h5>
          <CodeBlock language="bash">sudo apt install rayforge</CodeBlock>
        </div>
      </div>

      <div className="install-step">
        <div className="install-step-number">3</div>
        <div className="install-step-content">
          <h5>Launch Rayforge</h5>
          <p>
            Launch Rayforge from your application menu or by running:
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
      <h4>Flathub Package</h4>
      <p>
        The Flathub package is the easiest way to get started with Rayforge
        on Linux.
      </p>

      <div className="install-step">
        <div className="install-step-number">1</div>
        <div className="install-step-content">
          <h5>Install from Flathub</h5>
          <p>
            <a
              href="https://flathub.org/apps/org.rayforge.rayforge"
              target="_blank"
              rel="noopener noreferrer"
            >
              <img
                alt="Get it from Flathub"
                src="/images/flathub-badge.svg"
                height="55"
              />
            </a>
          </p>
          <p>Or install via command line:</p>
          <CodeBlock language="bash">
            flatpak install flathub org.rayforge.rayforge
          </CodeBlock>
        </div>
      </div>

      <div className="install-step">
        <div className="install-step-number">2</div>
        <div className="install-step-content">
          <h5>Launch Rayforge</h5>
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
      <h4>Snap Package</h4>
      <p>
        The Snap package works on most Linux distributions and includes all
        dependencies in a sandboxed environment.
      </p>

      <div className="install-step">
        <div className="install-step-number">1</div>
        <div className="install-step-content">
          <h5>Install Rayforge</h5>
          <CodeBlock language="bash">sudo snap install rayforge</CodeBlock>
        </div>
      </div>

      <div className="install-step">
        <div className="install-step-number">2</div>
        <div className="install-step-content">
          <h5>Grant Permissions (Important!)</h5>
          <Admonition type="warning" title="Permissions Required">
            The Snap version runs in a sandbox and requires manual
            permission grants for hardware access.
          </Admonition>

          <p>
            <strong>For USB Serial Port Access:</strong>
          </p>
          <CodeBlock language="bash">
            {`# Enable experimental hotplug support (one-time setup)
sudo snap set system experimental.hotplug=true

# Connect your laser via USB, then grant access
sudo snap connect rayforge:serial-port`}
          </CodeBlock>

          <p>
            <strong>For Camera Access:</strong>
          </p>
          <CodeBlock language="bash">
            sudo snap connect rayforge:camera
          </CodeBlock>
        </div>
      </div>

      <div className="install-step">
        <div className="install-step-number">3</div>
        <div className="install-step-content">
          <h5>Verify Permissions</h5>
          <CodeBlock language="bash">
            snap connections rayforge
          </CodeBlock>
          <p>
            Look for <code>serial-port</code> in the list. If it shows
            "connected", you're ready to go.
          </p>
        </div>
      </div>
    </div>
  );
}

function LinuxPixiInstall() {
  return (
    <div className="install-section">
      <h4>Pixi (Developer Installation)</h4>
      <p>
        <a
          href="https://pixi.sh"
          target="_blank"
          rel="noopener noreferrer"
        >
          Pixi
        </a>{' '}
        is a fast package manager for Python projects. This method is
        recommended for developers who want to contribute to Rayforge or
        run the latest development version.
      </p>

      <Admonition type="info" title="Linux Only">
        Pixi installation is currently only available on Linux.
      </Admonition>

      <div className="install-step">
        <div className="install-step-number">1</div>
        <div className="install-step-content">
          <h5>Install Pixi</h5>
          <p>
            Install Pixi using the official installer:
          </p>
          <CodeBlock language="bash">
            curl -fsSL https://pixi.sh/install.sh | bash
          </CodeBlock>
          <p>
            After installation, restart your shell or run:
          </p>
          <CodeBlock language="bash">source ~/.bashrc</CodeBlock>
        </div>
      </div>

      <div className="install-step">
        <div className="install-step-number">2</div>
        <div className="install-step-content">
          <h5>Clone the Repository</h5>
          <CodeBlock language="bash">
            {`git clone https://github.com/barebaric/rayforge.git
cd rayforge`}
          </CodeBlock>
        </div>
      </div>

      <div className="install-step">
        <div className="install-step-number">3</div>
        <div className="install-step-content">
          <h5>Install System Dependency</h5>
          <p>Install the required Gtk/Adwaita package:</p>
          <CodeBlock language="bash">sudo apt install gir1.2-adw-1</CodeBlock>
        </div>
      </div>

      <div className="install-step">
        <div className="install-step-number">4</div>
        <div className="install-step-content">
          <h5>Install Dependencies</h5>
          <p>
            Pixi will automatically create a virtual environment and
            install all dependencies:
          </p>
          <CodeBlock language="bash">pixi install</CodeBlock>
        </div>
      </div>

      <div className="install-step">
        <div className="install-step-number">5</div>
        <div className="install-step-content">
          <h5>Add User to dialout Group</h5>
          <p>Required for serial port access:</p>
          <CodeBlock language="bash">
            sudo usermod -a -G dialout $USER
          </CodeBlock>
          <p>
            <strong>Important:</strong> Log out and log back in for this
            change to take effect.
          </p>
        </div>
      </div>

      <div className="install-step">
        <div className="install-step-number">6</div>
        <div className="install-step-content">
          <h5>Run Rayforge</h5>
          <CodeBlock language="bash">pixi run rayforge</CodeBlock>
        </div>
      </div>

      <div className="install-troubleshoot">
        <h5>Useful Pixi Commands</h5>
        <details>
          <summary>Available development commands</summary>
          <div className="install-troubleshoot-content">
            <ul>
              <li>
                <code>pixi run test</code> - Run the test suite
              </li>
              <li>
                <code>pixi run uitest</code> - Run UI tests
              </li>
              <li>
                <code>pixi run lint</code> - Run linting and static analysis
              </li>
              <li>
                <code>pixi run format</code> - Format code with ruff
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
      <h4>Install from Source</h4>
      <p>
        For developers and advanced users who want to run the latest
        development version or contribute to the project.
      </p>

      <div className="install-step">
        <div className="install-step-number">1</div>
        <div className="install-step-content">
          <h5>Install System Dependencies</h5>
          <p>On Debian/Ubuntu:</p>
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
          <h5>Install Rayforge from PyPI</h5>
          <CodeBlock language="bash">pip3 install rayforge</CodeBlock>
          <Admonition type="note">
            Package names may differ on other distributions. Refer to your
            distribution's documentation.
          </Admonition>
        </div>
      </div>

      <div className="install-step">
        <div className="install-step-number">3</div>
        <div className="install-step-content">
          <h5>Add User to dialout Group</h5>
          <p>Required for serial port access:</p>
          <CodeBlock language="bash">
            sudo usermod -a -G dialout $USER
          </CodeBlock>
          <p>
            <strong>Important:</strong> Log out and log back in for this
            change to take effect.
          </p>
        </div>
      </div>
    </div>
  );
}

function WindowsInstall({ version }) {
  const downloadUrl = `https://github.com/barebaric/rayforge/releases/download/${version}/rayforge-v${version}-installer.exe`;
  return (
    <div className="install-section">
      <h4>Windows Installation</h4>

      <div className="install-step">
        <div className="install-step-number">1</div>
        <div className="install-step-content">
          <h5>Download the Installer</h5>
          <p>
            <a
              href={downloadUrl}
              target="_blank"
              rel="noopener noreferrer"
            >
              <strong>Download Rayforge v{version}</strong>
            </a>
          </p>
        </div>
      </div>

      <div className="install-step">
        <div className="install-step-number">2</div>
        <div className="install-step-content">
          <h5>Run the Installer</h5>
          <p>
            Run the downloaded installer and follow the on-screen
            instructions.
          </p>
          <Admonition type="tip">
            If you encounter permission errors, try running the installer
            as Administrator (right-click → Run as Administrator).
          </Admonition>
        </div>
      </div>

      <div className="install-step">
        <div className="install-step-number">3</div>
        <div className="install-step-content">
          <h5>Launch Rayforge</h5>
          <p>
            Launch Rayforge from the Start Menu or Desktop shortcut.
          </p>
        </div>
      </div>
    </div>
  );
}

function MacosInstall() {
  return (
    <div className="install-section">
      <h4>macOS Installation</h4>

      <Admonition type="info" title="Community Support">
        There are currently no official macOS builds. However, Rayforge
        may run from source using the pip installation method. Community
        contributions for macOS packaging are welcome!
      </Admonition>

      <div className="install-step">
        <div className="install-step-number">1</div>
        <div className="install-step-content">
          <h5>Install Dependencies</h5>
          <p>
            Using Homebrew, install the required dependencies:
          </p>
          <CodeBlock language="bash">
            brew install python3 gtk4 adwaita-icon-theme libvips opencv
          </CodeBlock>
        </div>
      </div>

      <div className="install-step">
        <div className="install-step-number">2</div>
        <div className="install-step-content">
          <h5>Install Rayforge</h5>
          <CodeBlock language="bash">pip3 install rayforge</CodeBlock>
        </div>
      </div>

      <div className="install-step">
        <div className="install-step-number">3</div>
        <div className="install-step-content">
          <h5>Install USB Drivers (if needed)</h5>
          <p>
            If your laser controller uses a CH340/CH341 chipset, install
            the drivers:
          </p>
          <CodeBlock language="bash">
            brew install --cask wch-ch34x-usb-serial-driver
          </CodeBlock>
        </div>
      </div>
    </div>
  );
}

function VerifyInstall({ os }) {
  return (
    <div className="install-section">
      <h4>Verify Installation</h4>

      <div className="install-step">
        <div className="install-step-number">✓</div>
        <div className="install-step-content">
          <h5>Launch Rayforge</h5>
          <p>
            Launch Rayforge from your application menu or terminal. You
            should see the main window with the canvas and toolbar.
          </p>
        </div>
      </div>

      <div className="install-step">
        <div className="install-step-number">✓</div>
        <div className="install-step-content">
          <h5>Check Version</h5>
          <p>
            Check <strong>Help → About</strong> to confirm the installed
            version.
          </p>
        </div>
      </div>

      <div className="install-step">
        <div className="install-step-number">✓</div>
        <div className="install-step-content">
          <h5>Connect Your Machine</h5>
          <p>
            Connect your laser controller via USB and verify that the
            serial port appears in Rayforge's machine settings.
          </p>
          {os === 'linux' && (
            <Admonition type="tip">
              On Linux, laser controllers typically appear as{' '}
              <code>/dev/ttyUSB0</code> or <code>/dev/ttyACM0</code>.
            </Admonition>
          )}
          {os === 'windows' && (
            <Admonition type="tip">
              On Windows, check Device Manager under "Ports (COM & LPT)"
              to find your COM port number.
            </Admonition>
          )}
        </div>
      </div>
    </div>
  );
}

function Troubleshooting({ os, linuxMethod }) {
  return (
    <div className="install-section">
      <h4>Troubleshooting</h4>
      {os === 'linux' && linuxMethod === 'snap' && (
        <div className="install-troubleshoot">
          <details>
            <summary>Permission issues?</summary>
            <div className="install-troubleshoot-content">
              <p>
                If your device isn't detected, see the{' '}
                <a href="../troubleshooting/snap-permissions">
                  Snap Permissions Guide
                </a>.
              </p>
            </div>
          </details>
          <details>
            <summary>Serial port still not working?</summary>
            <div className="install-troubleshoot-content">
              <ol>
                <li>
                  <strong>Replug the USB device:</strong> Unplug your laser
                  controller, wait 5 seconds, plug it back in.
                </li>
                <li>
                  <strong>Restart Rayforge:</strong> Close completely and
                  relaunch.
                </li>
                <li>
                  <strong>Check the device exists:</strong>
                  <CodeBlock language="bash">
                    ls -l /dev/ttyUSB* /dev/ttyACM*
                  </CodeBlock>
                </li>
                <li>
                  <strong>Reinstall if needed:</strong>
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
            <summary>Permission denied errors?</summary>
            <div className="install-troubleshoot-content">
              <p>
                If you get "permission denied" errors when accessing the
                serial port, add your user to the <code>dialout</code> group:
              </p>
              <CodeBlock language="bash">
                sudo usermod -a -G dialout $USER
              </CodeBlock>
              <p>Then log out and back in for changes to take effect.</p>
            </div>
          </details>
        </div>
      )}
      {os === 'windows' && (
        <div className="install-troubleshoot">
          <details>
            <summary>Installer crashing or not installing?</summary>
            <div className="install-troubleshoot-content">
              <p>
                If the installer crashes or fails to install, you may be
                missing the Microsoft Visual C++ Redistributable. Download
                and install it from Microsoft:
              </p>
              <p>
                <a
                  href="https://aka.ms/vc14/vc_redist.x64.exe"
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  <strong>Download C++ Redistributable v14</strong>
                </a>
              </p>
              <p>
                After installing the redistributable, run the Rayforge
                installer again.
              </p>
            </div>
          </details>
          <details>
            <summary>Driver issues?</summary>
            <div className="install-troubleshoot-content">
              <p>
                If your laser controller isn't detected, check Device
                Manager under <strong>Ports (COM & LPT)</strong> to confirm
                the COM port number. You may need to install CH340 or CP2102
                drivers for your USB-to-serial adapter.
              </p>
            </div>
          </details>
          <details>
            <summary>Antivirus blocking connection?</summary>
            <div className="install-troubleshoot-content">
              <p>
                Some antivirus software may block Rayforge from accessing USB
                devices. Try adding an exception if you experience connection
                issues.
              </p>
            </div>
          </details>
        </div>
      )}
      {os === 'macos' && (
        <div className="install-troubleshoot">
          <details>
            <summary>USB device not detected?</summary>
            <div className="install-troubleshoot-content">
              <p>
                If your device isn't detected, you may need to approve the
                USB device in <strong>System Settings → Privacy &
                Security</strong>.
              </p>
            </div>
          </details>
          <details>
            <summary>Driver issues?</summary>
            <div className="install-troubleshoot-content">
              <p>
                Some USB-to-serial adapters require additional drivers. Check
                if your adapter uses CH340 or CP2102 chips and install the
                appropriate driver.
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
      <h4>Need Help?</h4>
      <p>
        For additional help, report an issue on{' '}
        <a href="https://github.com/barebaric/rayforge/issues">GitHub</a> or
        join our <a href="https://discord.gg/sTHNdTtpQJ">Discord</a>.
      </p>
    </div>
  );
}

export default function InstallGuide() {
  const hashState = getInitialStateFromHash();
  const [selectedOs, setSelectedOs] = useState(() => hashState.os || detectOs());
  const [linuxMethod, setLinuxMethod] = useState(() => hashState.method || 'snap');

  useEffect(() => {
    const detected = detectOs();
    const hashState = getInitialStateFromHash();
    if (hashState.os) {
      setSelectedOs(hashState.os);
      if (hashState.method) {
        setLinuxMethod(hashState.method);
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
        {selectedOs === 'windows' && <WindowsInstall version={VERSION} />}
        {selectedOs === 'macos' && <MacosInstall />}

        <VerifyInstall os={selectedOs} />
        <Troubleshooting os={selectedOs} linuxMethod={linuxMethod} />
        <NeedHelp />
      </div>
    </div>
  );
}
