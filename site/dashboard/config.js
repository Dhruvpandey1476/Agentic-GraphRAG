// Where the live-query API lives.
//
// Empty string = same origin, which is what `python server.py` gives you.
// Set it to a deployed backend when the pages are hosted statically (GitHub
// Pages, Netlify, S3) and the Flask app runs somewhere else:
//
//   window.API_BASE = "https://agentic-graphrag.onrender.com";
//
// The backend must then allow this page's origin — set ALLOWED_ORIGIN on it
// to the static site's URL, or the browser will block the request with a CORS
// error. The benchmark dashboard needs none of this: it reads JSON files that
// ship beside it.
window.API_BASE = "";

// This copy is the statically hosted mirror (GitHub Pages). The benchmark
// dashboard is fully functional here — it only reads the JSON beside it. The
// live-query page needs a running backend: set API_BASE above to a deployed
// server (see DEPLOY.md), or run `python server.py` locally and use
// http://localhost:5000 instead.
