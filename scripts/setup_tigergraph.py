"""
One-command TigerGraph setup: connect -> install schema -> install
queries -> load the corpus -> verify.

    python -m scripts.setup_tigergraph              # everything
    python -m scripts.setup_tigergraph --check      # connection test only
    python -m scripts.setup_tigergraph --schema     # schema only
    python -m scripts.setup_tigergraph --queries    # queries only
    python -m scripts.setup_tigergraph --load       # data only
    python -m scripts.setup_tigergraph --verify     # run sample queries

Every step is idempotent, so re-running after a failure is safe.

Reads connection details from .env (see .env.example). If you haven't
provisioned an instance yet, run --check and it prints the exact steps.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

SCHEMA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "schema", "schema.gsql")
QUERIES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "schema", "queries.gsql")

PROVISION_HELP = """
No TigerGraph connection configured. To provision a free Savanna instance:

  1. Go to https://tgcloud.io and sign in.
  2. Click "Create Solution".
       Type:     TigerGraph Savanna (free tier / hackathon credits)
       Region:   whichever is closest to you
       Name:     anything, e.g. "agentic-graphrag"
     Provisioning takes ~5 minutes.
  3. Open the solution -> copy its "Domain"
     (looks like:  abcd1234.i.tgcloud.io)
  4. In the solution: Admin -> Management -> Create Secret.
     Give it any alias, copy the generated secret string.
  5. Put both into your .env:

       TG_HOST=https://abcd1234.i.tgcloud.io
       TG_GRAPH=HackathonGraph
       TG_USERNAME=tigergraph
       TG_PASSWORD=<the password you set at signup>
       TG_SECRET=<the secret from step 4>

  6. Re-run:  python -m scripts.setup_tigergraph

Running Community Edition locally instead? Download from
https://dl.tigergraph.com, start it, then use:

       TG_HOST=http://localhost
       TG_GS_PORT=14240
       TG_RESTPP_PORT=9000
       TG_USERNAME=tigergraph
       TG_PASSWORD=tigergraph
"""


def connect():
    if not config.TG_HOST or not (config.TG_SECRET or config.TG_PASSWORD):
        print(PROVISION_HELP)
        return None
    from src.tigergraph_client import RealGraph
    try:
        g = RealGraph()
        print(f"[ok] connected to {config.TG_HOST} (graph={config.TG_GRAPH})")
        return g
    except Exception as e:
        print(f"[FAIL] could not connect: {type(e).__name__}: {e}\n")
        print("Common causes:")
        print("  * TG_HOST missing the https:// prefix, or including a port")
        print("  * the solution is still provisioning, or is paused/stopped")
        print("  * TG_SECRET copied with surrounding whitespace or quotes")
        print("  * Community Edition: TG_GS_PORT/TG_RESTPP_PORT not set (14240/9000)")
        return None


def install_schema(graph):
    print(f"[..] installing schema from {os.path.basename(SCHEMA)}")
    with open(SCHEMA, encoding="utf-8") as f:
        script = f.read()
    # DROP GRAPH fails loudly on a fresh instance where nothing exists yet.
    # That's expected, not an error, so the drop is issued separately and
    # its failure is ignored rather than aborting the whole install.
    head, _, body = script.partition("CREATE GRAPH HackathonGraph()")
    if head.strip():
        try:
            graph.conn.gsql(head)
        except Exception as e:
            print(f"     (drop skipped: {e})")
    out = graph.conn.gsql("CREATE GRAPH HackathonGraph()\n" + body)
    print(out[-1500:] if isinstance(out, str) else out)
    print("[ok] schema installed")


def install_queries(graph):
    print(f"[..] installing + compiling queries from {os.path.basename(QUERIES)}")
    print("     (INSTALL QUERY compiles to C++ — this takes 1-3 minutes)")
    t0 = time.time()
    with open(QUERIES, encoding="utf-8") as f:
        out = graph.conn.gsql(f.read())
    print(out[-2500:] if isinstance(out, str) else out)
    print(f"[ok] queries installed in {time.time()-t0:.0f}s")
    print(f"     installed: {graph.query_installed()}")


def load_data(graph, limit=None, structured_only=False):
    print(f"[..] loading corpus into TigerGraph (backend={graph.backend})")
    from src.ingestion import build_graph
    build_graph.build(limit=limit, do_embed=not structured_only,
                      structured_only=structured_only)
    print("[ok] data loaded")


def verify(graph):
    print("\n[..] verifying with live queries")
    checks = []

    try:
        n = graph.conn.getVertexCount("OlympicEvent")
        checks.append(("OlympicEvent vertices", n, n > 1000))
    except Exception as e:
        checks.append(("OlympicEvent vertices", f"ERROR {e}", False))

    try:
        chrono = graph.games_chronology()
        checks.append(("GamesEdition vertices", len(chrono), len(chrono) > 5))
    except Exception as e:
        checks.append(("GamesEdition vertices", f"ERROR {e}", False))

    try:
        r = graph.filter_olympic_events(discipline="Biathlon", games_id="2018-winter",
                                        min_competitors=73)
        checks.append(("aggregation query (expect 5)", len(r), len(r) == 5))
    except Exception as e:
        checks.append(("aggregation query", f"ERROR {e}", False))

    try:
        n = graph.conn.getVertexCount("Chunk")
        checks.append(("Chunk vertices", n, True))
    except Exception as e:
        checks.append(("Chunk vertices", f"ERROR {e}", False))

    print()
    ok = True
    for name, value, passed in checks:
        print(f"  {'PASS' if passed else 'FAIL'}  {name:<32} {value}")
        ok = ok and passed
    print(f"\n{'All checks passed.' if ok else 'Some checks failed — see above.'}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--schema", action="store_true")
    ap.add_argument("--queries", action="store_true")
    ap.add_argument("--load", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--structured-only", action="store_true")
    args = ap.parse_args()

    graph = connect()
    if graph is None:
        sys.exit(1)
    if args.check:
        return

    everything = not any([args.schema, args.queries, args.load, args.verify])
    if everything or args.schema:
        install_schema(graph)
    if everything or args.queries:
        install_queries(graph)
    if everything or args.load:
        load_data(graph, args.limit, args.structured_only)
    if everything or args.verify:
        verify(graph)


if __name__ == "__main__":
    main()
