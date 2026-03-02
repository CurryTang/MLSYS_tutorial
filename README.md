# MLSYS Tutorial Notes Site

React + Vite frontend for browsing Markdown notes copied into `notes/`.

## Local development

```bash
npm install
npm run dev
```

## Build

```bash
npm run build
npm run preview
```

## GitHub Pages deployment

This repo includes `.github/workflows/deploy-pages.yml`.

1. Push this project to GitHub.
2. In GitHub repo settings, enable **Pages** and choose **GitHub Actions** as the source.
3. Push to `main` to auto-deploy.

Vite base path is inferred from `GITHUB_REPOSITORY` in CI.
If you need a custom path, set `VITE_BASE_PATH` during build.

## Notes content

All markdown files in `notes/` are auto-indexed by the frontend using `import.meta.glob`.
Any `.md` file added there will appear in the UI after rebuild.
