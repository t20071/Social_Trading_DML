# Self-Trading Explorer (web app)

An interactive, self-contained web page for the results of this project. It includes:
- a replay of the analysis run, stage by stage;
- animated charts of the data, the uncertainty, the week decomposition, heterogeneity, robustness and feedback tests.

**Plug and play.** Everything lives in this folder. Nothing else in the repository imports, references or depends on it. To remove the web app, delete `webapp/`.

## Files

| File | Purpose |
|---|---|
| `index.html` | the page |
| `styles.css` | design (dark by default, follows the system light/dark setting) |
| `app.js` | all charts and interactions, plain JavaScript and SVG, no libraries |
| `data.js` | the results bundled for the page; generated, do not edit |
| `export_data.py` | regenerates `data.js` from `../results/` (Python standard library only) |
| `vercel.json`, `.vercelignore` | settings for a Vercel deployment |

## Run locally

Double-click `index.html`. It works straight from the file system: no server, no build, no install. The fonts come from Google Fonts when online; offline, a system font is used.

To serve it instead:

```bash
python -m http.server 8000 --directory webapp
# open http://localhost:8000
```

## Refresh the data after a new pipeline run

```bash
python run_pipeline.py            # writes results/
python webapp/export_data.py      # rebuilds webapp/data.js from results/
```

## Deploy to Vercel

`data.js` is git-ignored (`webapp/.gitignore`), so the results are not on GitHub. Deploy from your machine, which uploads the local `data.js`:

```bash
npm i -g vercel          # once
cd webapp
vercel                   # preview; framework "Other", no build command
vercel --prod            # production URL
```

A deployment from the GitHub repository alone would have no `data.js`, and the page would say "No data found".

## A note on publishing

`data.js` contains aggregated results only (estimates, intervals, per-week summaries); there are no trader-level rows. A Vercel URL is public, so anyone with the link can see those results.
