# Build Profiles

Date locked: 2026-04-08

This directory defines selective build profiles for Leopardi data preparation.

The goal is to allow:

- building only the exact core
- building only formula specialists
- building only handwriting specialists
- building only finetune repair packs
- building the full stack

This is mandatory for ephemeral machines and iterative experiments.

## Files

- `build-profiles.md`
  - profile definitions and intended usage
- `profile-registry.csv`
  - compact machine-readable profile index
