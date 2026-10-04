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
