# Getting the Code

This guide covers how to obtain the Rayforge source code for development.

## Fork the Repository

Fork the [Rayforge repository](https://github.com/barebaric/rayforge) on GitHub to create your own copy where you can make changes.

## Clone Your Fork

```bash
git clone https://github.com/YOUR_USERNAME/rayforge.git
cd rayforge
```

## Add Upstream Repository

Add the original repository as an upstream remote to keep track of changes:

```bash
git remote add upstream https://github.com/barebaric/rayforge.git
```

## Verify the Repository

Check that the remotes are configured correctly:

```bash
git remote -v
```

You should see both your fork (origin) and the upstream repository.

## Next Steps

After getting the code, continue with [Setup](setup.md) to configure your development environment.