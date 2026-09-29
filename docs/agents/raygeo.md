# Raygeo (Rust/PyO3 geometry library)

Even though Raygeo is installed as a regular pip dependency, we own it. If the
root cause of an issue is in Raygeo, you should fix it there instead of
building a workaround.

Source repository: https://github.com/barebaric/raygeo

## Testing with a local Raygeo checkout

Attention: Running "pixi run", even to lint, reinstalls from PyPi.

`scripts/pixi-raygeo.sh` wraps any pixi command with a
`dependency-override` that uses a local raygeo checkout. The project's
real `pixi.toml`/`pixi.lock` are never permanently modified.

```bash
ln -s /path/to/raygeo external/raygeo    # one-time symlink (external/ is gitignored)
scripts/pixi-raygeo.sh run rayforge      # run against local raygeo
scripts/pixi-raygeo.sh run test          # test against local raygeo
scripts/pixi-raygeo.sh shell             # activate a shell with local raygeo
```

After editing raygeo Rust or Python source, rebuild it with:

```bash
scripts/rebuild-raygeo.sh                # clear uv cache + rebuild raygeo
```

To go back to the PyPI raygeo, just use `pixi run rayforge` without the
wrapper (or any other pixi command).
