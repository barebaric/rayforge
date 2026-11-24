---
description: "Build and test the Rayforge MSIX package for Windows."
---

# Windows MSIX Package

Rayforge can be packaged as an MSIX app for per-user installation and Microsoft Store distribution.
The package uses the existing PyInstaller bundle, so build the normal Windows executable first.

## Prerequisites

- Complete the [Windows development setup](setup.md).
- Install the Windows SDK, which provides `makeappx.exe` and `signtool.exe`.

## Build

From the MSYS2 UCRT64 shell, build the executable and then the package:

```bash
bash scripts/win/win_build.sh 1.15.0
bash scripts/win/win_msix.sh 1.15.0
```

The package is written to `dist/rayforge-v1.15.0-x64.msix`. The script renders
`scripts/win/msix/AppxManifest.xml.in` with the package identity and version, and generates the
required PNG assets.

## Configuration

The build script reads these environment variables:

| Variable                      | Description                                                                                      |
| ----------------------------- | ------------------------------------------------------------------------------------------------ |
| `MSIX_IDENTITY_NAME`          | Package identity name from Partner Center. Defaults to `Rayforge.Dev` for local builds.          |
| `MSIX_PUBLISHER`              | Publisher subject from Partner Center, for example `CN=...`. Must match the signing certificate. |
| `MSIX_PUBLISHER_DISPLAY_NAME` | Publisher name shown by Windows.                                                                 |
| `MSIX_VERSION`                | Optional four-part package version. Defaults to the build version.                               |
| `MSIX_CERT_PFX`               | Optional PFX file used to sign the package.                                                      |
| `MSIX_CERT_PASSWORD`          | Password for `MSIX_CERT_PFX`.                                                                    |
| `MSIX_TIMESTAMP_URL`          | Optional RFC 3161 timestamp server used while signing.                                           |

For Microsoft Store submissions, use the identity values shown in Partner Center. Packages uploaded
to the Store do not need a CA-trusted certificate; the Store re-signs them. For sideloading outside
the Store, sign with a certificate that the target devices trust.

The CI workflow builds, signs, and test-installs the MSIX on every Windows build. Store identity
values are read from the repository variables `MSIX_IDENTITY_NAME`, `MSIX_PUBLISHER`, and
`MSIX_PUBLISHER_DISPLAY_NAME`. A real signing certificate can be supplied as the secrets
`MSIX_CERT_PFX_BASE64` and `MSIX_CERT_PASSWORD`.

## Behavior Under MSIX

MSIX builds keep the application's self-updater disabled because the Store or App Installer owns
package updates. File associations are declared in the manifest, but Windows requires the user to
choose Rayforge as the default handler for shared file types.
