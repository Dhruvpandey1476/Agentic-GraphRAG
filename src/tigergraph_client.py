"""
Graph backend. Two implementations behind ONE interface:

  RealGraph  — pyTigerGraph against Savanna or Community Edition.
  MockGraph  — in-memory, backed by outputs/graph_store.json, so the
               repo runs end to end before any instance is provisioned.

Both expose exactly the same methods (see BaseGraph below). This matters:
the original version let callers reach into `graph.store[...]` directly,
which only MockGraph has — meaning RAG and similarity search silently
returned zero results the moment you pointed it at a real TigerGraph.
Every access now goes through the interface, so switching backends is a
config change and nothing else.

Backend selection (get_graph): TG_HOST + (TG_SECRET or TG_PASSWORD) and
FORCE_MOCK_GRAPH unset -> RealGraph. Fallback to MockGraph is loud, never
silent, because a benchmark that quietly stopped using TigerGraph is
worse than one that fails.
"""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.ingestion.infobox import normalize, contains_phrase


class BaseGraph:
    """The contract every pipeline and agent codes against."""

    backend = "base"

    # --- retrieval ---
    def vector_search(self, query_emb, k=5):
        """-> [(score, chunk_id, chunk_dict)], best first."""
        raise NotImplementedError

    def all_chunk_count(self) -> int:
        raise NotImplementedError

    def chunks_for_doc(self, doc_id) -> list:
        raise NotImplementedError

    def entity_lookup(self, name_substr) -> list:
        raise NotImplementedError

    def k_hop_neighborhood(self, seed_id, hops=2) -> list:
        raise NotImplementedError

    def entity_supporting_chunks(self, seed_id) -> list:
        raise NotImplementedError

    # --- structured Olympic layer ---
    def filter_olympic_events(self, **kw) -> list:
        raise NotImplementedError

    def games_chronology(self) -> dict:
        raise NotImplementedError

    def known_disciplines(self) -> list:
        raise NotImplementedError


# ==========================================================================
#  Real TigerGraph
# ==========================================================================

class RealGraph(BaseGraph):
    """pyTigerGraph wrapper hitting the installed GSQL queries in
    schema/queries.gsql.

    Savanna vs Community Edition differ in exactly two ways that matter
    here, both handled in __init__: Savanna terminates TLS on 443 and
    wants tgCloud=True, whereas Community Edition exposes RESTPP on 9000
    and GSQL on 14240 over plain HTTP.
    """

    backend = "tigergraph"

    def __init__(self, verify=True):
        import pyTigerGraph as tg

        host = config.TG_HOST.rstrip("/")
        is_cloud = host.startswith("https://") or ".tgcloud.io" in host

        kwargs = {
            "host": host,
            "graphname": config.TG_GRAPH,
            "username": config.TG_USERNAME,
            "password": config.TG_PASSWORD or "tigergraph",
            "tgCloud": is_cloud,
        }
        if config.TG_SECRET:
            kwargs["gsqlSecret"] = config.TG_SECRET
        if config.TG_RESTPP_PORT:
            kwargs["restppPort"] = config.TG_RESTPP_PORT
        if config.TG_GS_PORT:
            kwargs["gsPort"] = config.TG_GS_PORT

        self.conn = tg.TigerGraphConnection(**kwargs)

        # Token auth is preferred (and required on Savanna for RESTPP).
        try:
            if config.TG_SECRET:
                self.conn.apiToken = self.conn.getToken(config.TG_SECRET)
            else:
                self.conn.apiToken = self.conn.getToken()
        except Exception as e:
            # Community Edition with auth disabled has no token endpoint;
            # that's fine, keep going and let verify() decide.
            print(f"[tigergraph] token auth unavailable ({e}); continuing without a token")

        if verify:
            self.conn.echo()   # raises if unreachable / misconfigured

        self._chrono_cache = None
        self._disciplines_cache = None

    # ---------------------------------------------------------- admin

    def run_gsql(self, script_path):
        with open(script_path, encoding="utf-8") as f:
            return self.conn.gsql(f.read())

    def query_installed(self) -> list:
        try:
            return [q["queryName"] for q in self.conn.getInstalledQueries()]
        except Exception:
            return []

    # ---------------------------------------------------------- retrieval

    def vector_search(self, query_emb, k=5):
        """Native TigerGraph vector search over Chunk.embedding.

        Uses the installed `chunkVectorSearch` GSQL query, which calls
        TigerGraph's built-in vector index. Falls back to pulling chunk
        vectors and scoring in Python only if that query isn't installed,
        so a partially-set-up instance degrades instead of erroring.
        """
        from src.embeddings import cosine_sim

        vec = query_emb if isinstance(query_emb, list) else (
            list(query_emb.values()) if isinstance(query_emb, dict) else query_emb.tolist()
        )
        if "chunkVectorSearch" in self.query_installed() and not isinstance(query_emb, dict):
            try:
                raw = self.conn.runInstalledQuery(
                    "chunkVectorSearch", {"queryVector": vec, "k": k}, timeout=60000)
                out = []
                for v in raw[0]["Result"]:
                    a = v["attributes"]
                    out.append((float(a.get("score", 0.0)), v["v_id"],
                                {"text": a.get("text", ""), "doc_id": a.get("doc_id", "")}))
                if out:
                    return out
            except Exception as e:
                print(f"[tigergraph] vector query failed, scoring locally: {e}")

        # Fallback: score locally over chunks fetched from the graph.
        chunks = self._fetch_all_chunks()
        scored = [(cosine_sim(query_emb, c["embedding"]), cid, c)
                  for cid, c in chunks.items() if c.get("embedding")]
        scored.sort(key=lambda x: -x[0])
        return scored[:k]

    def _fetch_all_chunks(self):
        if getattr(self, "_chunk_cache", None) is None:
            verts = self.conn.getVertices("Chunk")
            self._chunk_cache = {v["v_id"]: v["attributes"] for v in verts}
        return self._chunk_cache

    def all_chunk_count(self) -> int:
        try:
            return self.conn.getVertexCount("Chunk")
        except Exception:
            return len(self._fetch_all_chunks())

    def chunks_for_doc(self, doc_id) -> list:
        return [dict(chunk_id=cid, **c) for cid, c in self._fetch_all_chunks().items()
                if c.get("doc_id") == doc_id]

    def entity_lookup(self, name_substr):
        return self.conn.runInstalledQuery("entityLookup", {"nameSubstr": name_substr})

    def k_hop_neighborhood(self, seed_id, hops=2):
        return self.conn.runInstalledQuery("kHopNeighborhood", {"seed": seed_id, "hops": hops})

    def entity_supporting_chunks(self, seed_id):
        return self.conn.runInstalledQuery("entitySupportingChunks", {"seed": seed_id})

    # ---------------------------------------------------------- structured

    def filter_olympic_events(self, discipline=None, games_id=None, min_competitors=None,
                              venue_substr=None, date_substr=None, event_name_substr=None):
        params = {
            "discipline": discipline or "", "gamesId": games_id or "",
            "minCompetitors": min_competitors if min_competitors is not None else -1,
            "venueSubstr": venue_substr or "", "dateSubstr": date_substr or "",
            "eventNameSubstr": event_name_substr or "",
        }
        raw = self.conn.runInstalledQuery("filterOlympicEvents", params, timeout=60000)
        return [self._normalize_event(v) for v in raw[0]["Result"]]

    def games_chronology(self):
        if self._chrono_cache is not None:
            return self._chrono_cache
        raw = self.conn.runInstalledQuery("gamesChronology", {})
        chrono = {}
        for v in raw[0]["Result"]:
            a = v["attributes"]
            chrono[v["v_id"]] = {
                "year": a["year"], "season": a["season"], "index": a["edition_index"],
                "prev_id": a["prev_id"] or None, "next_id": a["next_id"] or None,
            }
        self._chrono_cache = chrono
        return chrono

    def known_disciplines(self):
        if self._disciplines_cache is None:
            try:
                verts = self.conn.getVertices("Discipline")
                self._disciplines_cache = sorted(v["v_id"] for v in verts)
            except Exception:
                self._disciplines_cache = []
        return self._disciplines_cache

    @staticmethod
    def _normalize_event(v):
        """Flattens a pyTigerGraph OlympicEvent vertex into the exact shape
        MockGraph produces (infobox.parse_olympic_event), so every caller
        works identically against either backend."""
        a = v["attributes"]
        medalists = []
        for color in ("gold", "silver", "bronze"):
            if a.get(f"{color}_name"):
                medalists.append({"medal": color, "name": a[f"{color}_name"],
                                  "noc": a.get(f"{color}_noc", "")})
        return {
            "doc_id": v["v_id"], "title": a.get("title", ""), "url": a.get("url", ""),
            "discipline": a.get("discipline", ""), "event_name": a.get("event_name", ""),
            "venue": a.get("venue", ""), "venues": a.get("venues", ""), "date": a.get("date", ""),
            "games_id": a.get("games_id", ""),
            "competitors": a.get("competitors") or None, "nations": a.get("nations") or None,
            "teams": a.get("teams") or None, "medalists": medalists,
            "win_value": a.get("win_value", ""),
        }

    # ---------------------------------------------------------- bulk load

    def load_olympic_events(self, events: list, chronology: dict, batch=1000):
        """Batched upserts. One-vertex-at-a-time over HTTP would be ~3,000
        round trips for this corpus; upsertVertices sends them in blocks."""
        games = [(gid, {"year": g["year"], "season": g["season"],
                        "edition_index": g["index"], "prev_id": g["prev_id"] or "",
                        "next_id": g["next_id"] or ""})
                 for gid, g in chronology.items()]
        self._upsert_batched("GamesEdition", games, batch)

        seq = [(gid, g["next_id"], {}) for gid, g in chronology.items() if g["next_id"]]
        if seq:
            self.conn.upsertEdges("GamesEdition", "GAMES_SEQUENCE", "GamesEdition", seq)

        disciplines = sorted({e["discipline"] for e in events if e.get("discipline")})
        self._upsert_batched("Discipline", [(d, {}) for d in disciplines], batch)

        def medal(e, color):
            m = next((m for m in e["medalists"] if m["medal"] == color), None)
            return (m["name"], m["noc"]) if m else ("", "")

        rows = []
        for e in events:
            g_n, g_c = medal(e, "gold")
            s_n, s_c = medal(e, "silver")
            b_n, b_c = medal(e, "bronze")
            # The *_lc mirrors exist because GSQL's LIKE is case-sensitive
            # while the Python backend matches case-insensitively. Writing
            # them at load time is what makes the two backends return
            # identical results for the same question (see schema.gsql).
            venue_all = f"{e.get('venue', '')} {e.get('venues', '')}".strip()
            rows.append((e["doc_id"], {
                "title": e["title"], "url": e["url"], "discipline": e["discipline"],
                "discipline_lc": (e["discipline"] or "").lower(),
                "event_name": e["event_name"],
                "event_name_lc": (e["event_name"] or "").lower(),
                "venue": e.get("venue", ""),
                "venues": e.get("venues", ""), "venue_lc": venue_all.lower(),
                "date": e.get("date", ""), "date_lc": (e.get("date", "") or "").lower(),
                "games_id": e.get("games_id", ""),
                "competitors": e["competitors"] or 0, "nations": e["nations"] or 0,
                "teams": e["teams"] or 0,
                "gold_name": g_n, "gold_noc": g_c, "silver_name": s_n, "silver_noc": s_c,
                "bronze_name": b_n, "bronze_noc": b_c, "win_value": e.get("win_value", ""),
            }))
        self._upsert_batched("OlympicEvent", rows, batch)

        in_games = [(e["doc_id"], e["games_id"], {}) for e in events if e.get("games_id")]
        self._upsert_edges_batched("OlympicEvent", "IN_GAMES", "GamesEdition", in_games, batch)
        in_disc = [(e["doc_id"], e["discipline"], {}) for e in events if e.get("discipline")]
        self._upsert_edges_batched("OlympicEvent", "IN_DISCIPLINE", "Discipline", in_disc, batch)

    def load_documents_and_chunks(self, documents: list, chunks: list, batch=500):
        """documents: [(doc_id, attrs)], chunks: [(chunk_id, attrs)] where
        attrs['embedding'] is a dense list. Sparse TF-IDF dicts can't go in
        a TigerGraph LIST<DOUBLE>, so the caller must use a dense embedding
        provider (openai/ollama) when VECTOR_BACKEND=tigergraph."""
        self._upsert_batched("Document", documents, batch)
        self._upsert_batched("Chunk", chunks, batch)
        edges = [(c[1]["doc_id"], cid, {}) for cid, c in
                 ((c[0], c) for c in chunks) if c[1].get("doc_id")]
        self._upsert_edges_batched("Document", "HAS_CHUNK", "Chunk", edges, batch)

    def _upsert_batched(self, vtype, rows, batch):
        for i in range(0, len(rows), batch):
            self.conn.upsertVertices(vtype, rows[i:i + batch])

    def _upsert_edges_batched(self, src_t, etype, tgt_t, rows, batch):
        for i in range(0, len(rows), batch):
            self.conn.upsertEdges(src_t, etype, tgt_t, rows[i:i + batch])

    # single-item upserts kept for ingestion code that builds incrementally
    def upsert_vertex(self, v_type, v_id, attrs):
        self.conn.upsertVertex(v_type, v_id, attrs)

    def upsert_edge(self, src_type, src_id, edge_type, tgt_type, tgt_id, attrs=None):
        self.conn.upsertEdge(src_type, src_id, edge_type, tgt_type, tgt_id, attrs or {})


# ==========================================================================
#  In-memory mock
# ==========================================================================

class MockGraph(BaseGraph):
    """In-memory stand-in with identical method signatures. Backed by
    outputs/graph_store.json. Two data layers:
      - generic Document/Chunk/Entity (RAG + free-form entity linking)
      - structured Olympic-event store (olympic_events / games_chronology)
        built once by src/ingestion/build_graph.py — this is what
        filter_olympic_events() computes exact answers against.
    """

    backend = "mock"

    def __init__(self, store_path=None):
        self.store_path = store_path or os.path.join(config.OUTPUT_DIR, "graph_store.json")
        if os.path.exists(self.store_path):
            with open(self.store_path, encoding="utf-8") as f:
                self.store = json.load(f)
        else:
            self.store = {}
        for k in ("entities", "chunks", "documents", "olympic_events", "games_chronology"):
            self.store.setdefault(k, {})
        for k in ("relations", "mentions"):
            self.store.setdefault(k, [])
        self._disciplines_cache = None

    def save(self):
        os.makedirs(os.path.dirname(self.store_path), exist_ok=True)
        with open(self.store_path, "w", encoding="utf-8") as f:
            json.dump(self.store, f)

    # ---- generic upserts ----

    def upsert_vertex(self, v_type, v_id, attrs):
        bucket = {"Entity": "entities", "Chunk": "chunks", "Document": "documents"}[v_type]
        self.store[bucket][v_id] = attrs

    def upsert_edge(self, src_type, src_id, edge_type, tgt_type, tgt_id, attrs=None):
        if edge_type == "RELATED_TO":
            self.store["relations"].append({"src": src_id, "tgt": tgt_id, **(attrs or {})})
        elif edge_type == "MENTIONS":
            self.store["mentions"].append({"chunk": src_id, "entity": tgt_id, **(attrs or {})})

    # ---- retrieval ----

    def vector_search(self, query_emb, k=5):
        from src.embeddings import cosine_sim
        scored = [(cosine_sim(query_emb, c["embedding"]), cid, c)
                  for cid, c in self.store["chunks"].items() if c.get("embedding")]
        scored.sort(key=lambda x: -x[0])
        return scored[:k]

    def all_chunk_count(self):
        return len(self.store["chunks"])

    def chunks_for_doc(self, doc_id):
        return [dict(chunk_id=cid, **c) for cid, c in self.store["chunks"].items()
                if c.get("doc_id") == doc_id]

    def entity_lookup(self, name_substr):
        name_substr = name_substr.lower()
        return [{"entity_id": eid, **e} for eid, e in self.store["entities"].items()
                if name_substr in e.get("name", "").lower()][:10]

    def k_hop_neighborhood(self, seed_id, hops=2):
        visited, frontier = {seed_id}, {seed_id}
        for _ in range(hops):
            nxt = set()
            for rel in self.store["relations"]:
                if rel["src"] in frontier and rel["tgt"] not in visited:
                    nxt.add(rel["tgt"])
                if rel["tgt"] in frontier and rel["src"] not in visited:
                    nxt.add(rel["src"])
            visited |= nxt
            frontier = nxt
        visited.discard(seed_id)
        return [{"entity_id": eid, **self.store["entities"].get(eid, {})} for eid in visited]

    def entity_supporting_chunks(self, seed_id):
        chunk_ids = {m["chunk"] for m in self.store["mentions"] if m["entity"] == seed_id}
        return [{"chunk_id": cid, **self.store["chunks"].get(cid, {})} for cid in chunk_ids]

    # ---- structured Olympic layer ----

    def load_olympic_events(self, events: list, chronology: dict):
        self.store["olympic_events"] = {e["doc_id"]: e for e in events}
        self.store["games_chronology"] = chronology

    def games_chronology(self):
        return self.store["games_chronology"]

    def known_disciplines(self):
        if self._disciplines_cache is None:
            self._disciplines_cache = sorted(
                {e["discipline"] for e in self.store["olympic_events"].values() if e.get("discipline")})
        return self._disciplines_cache

    def filter_olympic_events(self, discipline=None, games_id=None, min_competitors=None,
                              venue_substr=None, date_substr=None, event_name_substr=None):
        results = []
        for e in self.store["olympic_events"].values():
            if discipline and normalize(discipline) != normalize(e["discipline"]):
                continue
            if games_id and games_id != e["games_id"]:
                continue
            if min_competitors is not None:
                if e["competitors"] is None or e["competitors"] <= min_competitors:
                    continue
            if venue_substr and not contains_phrase(
                    e.get("venue", "") + " " + e.get("venues", ""), venue_substr):
                continue
            if date_substr and not contains_phrase(e.get("date", ""), date_substr):
                continue
            if event_name_substr and not contains_phrase(e.get("event_name", ""), event_name_substr):
                continue
            results.append(e)
        return results


def get_graph(verbose=True):
    """Factory. Returns RealGraph when TigerGraph looks configured and
    reachable, else MockGraph. A fallback always prints why — a run that
    quietly stopped using TigerGraph would invalidate the whole benchmark."""
    if config.FORCE_MOCK_GRAPH:
        if verbose:
            print("[graph] FORCE_MOCK_GRAPH=1 -> MockGraph")
        return MockGraph()

    if config.TG_HOST and (config.TG_SECRET or config.TG_PASSWORD):
        try:
            g = RealGraph()
            if verbose:
                print(f"[graph] connected to TigerGraph at {config.TG_HOST} "
                      f"(graph={config.TG_GRAPH})")
            return g
        except Exception as e:
            print(f"[graph] !! TigerGraph connection FAILED, falling back to MockGraph: {e}")
    elif verbose:
        print("[graph] no TG_HOST/credentials configured -> MockGraph "
              "(set them in .env to use TigerGraph)")
    return MockGraph()
