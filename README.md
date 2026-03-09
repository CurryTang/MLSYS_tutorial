# MLSYS Tutorial

This repository hosts the MLSYS tutorial notes site.

## Live Site

**GitHub Pages:** [https://currytang.github.io/MLSYS_tutorial/](https://currytang.github.io/MLSYS_tutorial/)

The site publishes the curated MLSYS tutorials from `notes/Mlsys/` and now supports both Chinese and English versions in the frontend reader.

## Repository Layout

- `notes/Mlsys/`: tutorial markdown files and local assets
- `src/`: React frontend for browsing and rendering the tutorials
- `docs/plans/`: design and implementation notes for repo changes

## Local Development

```bash
npm install
npm run dev
```

## Verification

```bash
npm test
npm run lint
npm run build
```

## Deployment Notes

GitHub Pages deployment is handled by `.github/workflows/deploy-pages.yml`.

Vite infers the production base path from `GITHUB_REPOSITORY` in CI. If needed, override it with `VITE_BASE_PATH` during the build.
