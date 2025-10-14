### Rayforge Flatpak
This is Flathub package of [Rayforge](https://github.com/barebaric/rayforge).

## Building
This package can be built locally by running
```
flatpak run org.flatpak.Builder --force-clean --sandbox --user --ccache --install-deps-from=flathub --repo=repo builddir org.rayforge.rayforge.yml
flatpak build-bundle repo rayforge.flatpak org.rayforge.rayforge
```

## Installation
```
flatpak --user install repo rayforge.flatpak
```
