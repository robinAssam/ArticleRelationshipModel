"""Pipeline smoke tests on the synthetic Lunar Medical Code.

The synthetic corpus is no longer part of evaluation; it lives on here as a
fixture that exercises every stage of the pipeline end to end. No sentence
encoder is loaded, so the tests run in a few seconds.

Run from the repo root:

    python -m unittest discover tests
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402

from src.data.atom_table import (  # noqa: E402
    LIST_FIELDS, _tied_rank_metrics, build_atom_table, build_edge_table,
    classify_atom, compute_tag_weights, drop_synthetic, evaluate_article_retrieval,
    evaluate_classifier_loo, filter_candidates, is_synthetic, only_synthetic,
    unique_tags_by_field,
)
from src.data.query import (  # noqa: E402
    parse_query_atoms, retrieve_for_text, text_to_query_atoms,
)
from src.data.hierarchy import (  # noqa: E402
    add_taxonomy_columns, build_reference_edges, evaluate_at_levels,
)
from src.data.reasoning import chained_reasoning, explain_chain  # noqa: E402

ANNOTATIONS = PROJECT_ROOT / "data" / "annotations"


class SyntheticSplitTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = build_atom_table(ANNOTATIONS)

    def test_split_partitions_corpus(self):
        real, synth = drop_synthetic(self.df), only_synthetic(self.df)
        self.assertGreater(len(real), 0)
        self.assertGreater(len(synth), 0)
        self.assertEqual(len(real) + len(synth), len(self.df))
        self.assertTrue(synth["article_id"].map(is_synthetic).all())
        self.assertFalse(real["article_id"].map(is_synthetic).any())

    def test_loo_excludes_synthetic_by_default(self):
        real = drop_synthetic(self.df)
        eval_df, summary = evaluate_classifier_loo(
            self.df, weights=compute_tag_weights(real), top_k=3,
        )
        self.assertEqual(summary["n_atoms"], len(real))
        self.assertFalse(eval_df["true_article"].map(is_synthetic).any())
        self.assertFalse(eval_df["top1_match"].dropna().map(is_synthetic).any())

    def test_loo_on_synthetic_only_needs_opt_in(self):
        synth = only_synthetic(self.df)
        with self.assertRaises(ValueError):
            evaluate_classifier_loo(synth)

    def test_level_eval_excludes_synthetic_by_default(self):
        eval_df, _ = evaluate_at_levels(self.df, top_k=3)
        self.assertFalse(eval_df["atom_id"].map(is_synthetic).any())


class SyntheticPipelineTest(unittest.TestCase):
    """End-to-end run of every pipeline stage on the synthetic corpus."""

    @classmethod
    def setUpClass(cls):
        cls.df = only_synthetic(build_atom_table(ANNOTATIONS)).reset_index(drop=True)
        cls.weights = compute_tag_weights(cls.df)
        cls.edges = build_edge_table(cls.df, weights=cls.weights)
        cls.ref_edges = build_reference_edges(add_taxonomy_columns(cls.df))

    def test_idf_weights_are_non_negative(self):
        self.assertTrue(self.weights)
        self.assertTrue(all(w >= 0 for w in self.weights.values()))

    def test_edges_are_cross_article(self):
        self.assertGreater(len(self.edges), 0)
        self.assertTrue((self.edges["article_a"] != self.edges["article_b"]).all())

    def test_classify_skips_self_and_ranks_descending(self):
        query = self.df.iloc[0].to_dict()
        out = classify_atom(query, self.df, weights=self.weights, top_k=3)
        scores = out["scores_by_atom"]
        self.assertNotIn(query["atom_id"], scores["atom_id"].values)
        self.assertEqual(len(out["top_matches"]), 3)
        self.assertTrue(scores["combined_score"].is_monotonic_decreasing)

    def test_filter_falls_back_to_full_corpus(self):
        query = {"actors": ["nobody in this corpus"]}
        self.assertEqual(len(filter_candidates(self.df, query)), len(self.df))

    def test_loo_runs_with_opt_in(self):
        _, summary = evaluate_classifier_loo(
            self.df, weights=self.weights, top_k=3, include_synthetic=True,
        )
        self.assertEqual(summary["n_atoms"], len(self.df))
        self.assertTrue(0.0 <= summary["top1_accuracy"] <= 1.0)

    def test_chained_reasoning_trace_is_auditable(self):
        query = self.df.iloc[0].to_dict()
        query["atom_id"] = "Q_test"
        result = chained_reasoning(
            query, self.df, self.ref_edges, self.edges,
            weights=self.weights, top_k=2, max_hops=1, min_edge_strength=0.0,
        )
        chain = result["chain"]
        anchors = [c for c in chain if c["hop"] == 0]
        self.assertEqual(len(anchors), 2)
        self.assertTrue(all(c["edge_kind"] == "classifier" for c in anchors))
        self.assertTrue(all(c["via"] for c in chain))
        self.assertEqual(len({c["atom_id"] for c in chain}), len(chain))
        self.assertTrue({c["article_id"] for c in chain} <= set(result["articles_touched"]))
        self.assertIn("Articles surfaced", explain_chain(result))


class ScoringTest(unittest.TestCase):
    """Normalised scoring and article-level ranking."""

    @classmethod
    def setUpClass(cls):
        cls.df = only_synthetic(build_atom_table(ANNOTATIONS)).reset_index(drop=True)
        cls.weights = compute_tag_weights(cls.df)

    def _query_copy(self, i=0):
        query = self.df.iloc[i].to_dict()
        query["atom_id"] = "Q_copy"
        return query

    def test_identical_atom_scores_one(self):
        query = self._query_copy()
        for norm in ("cosine", "query"):
            for alpha in (0.5, 0.75, 1.0):
                scores = classify_atom(query, self.df, weights=self.weights, alpha=alpha,
                                       normalize=norm)["scores_by_atom"]
                top = scores.iloc[0]
                self.assertEqual(top["atom_id"], self.df.iloc[0]["atom_id"])
                self.assertAlmostEqual(top["combined_score"], 1.0, places=3)

    def test_normalised_scores_in_unit_range(self):
        scores = classify_atom(self._query_copy(3), self.df, weights=self.weights)["scores_by_atom"]
        self.assertTrue(((scores["combined_score"] >= 0) & (scores["combined_score"] <= 1.0001)).all())

    def test_article_ranking_has_one_row_per_article(self):
        out = classify_atom(self._query_copy(), self.df, weights=self.weights)
        ranking = out["article_ranking"]
        self.assertEqual(len(ranking), self.df["article_id"].nunique())
        self.assertEqual(out["predicted_article"], ranking.iloc[0]["article_id"])
        self.assertTrue(ranking["article_score"].is_monotonic_decreasing)

    def test_unknown_mode_rejected(self):
        with self.assertRaises(ValueError):
            classify_atom(self._query_copy(), self.df, normalize="bogus")

    def test_ties_scored_by_expected_value(self):
        ranking = pd.DataFrame({"article_id": ["a", "b", "c", "d"],
                                "article_score": [0.9, 0.0, 0.0, 0.0]})
        m = _tied_rank_metrics(ranking, "c", ks=(1, 3))
        # c is tied with b and d at ranks 2-4
        self.assertAlmostEqual(m["rank"], 3.0)
        self.assertAlmostEqual(m["rr"], (1 / 2 + 1 / 3 + 1 / 4) / 3)
        self.assertAlmostEqual(m["hit@1"], 0.0)
        self.assertAlmostEqual(m["hit@3"], 2 / 3)

    def test_article_eval_flags_unanswerable_and_reports_random(self):
        eval_df, summary = evaluate_article_retrieval(self.df, include_synthetic=True)
        sizes = self.df.groupby("article_id").size()
        singletons = set(sizes[sizes == 1].index)
        self.assertEqual(summary["n_answerable"], len(self.df) - len(singletons))
        unanswerable = eval_df[~eval_df["answerable"]]
        self.assertTrue(set(unanswerable["true_article"]) <= singletons)
        self.assertTrue((unanswerable["rr"] == 0).all())
        self.assertGreater(summary["mrr"], summary["random_mrr"])


class _StubBlock:
    def __init__(self, text):
        self.type, self.text = "text", text


class _StubClient:
    """Stands in for anthropic.Anthropic: records the request, returns canned JSON."""

    def __init__(self, payload, stop_reason="end_turn"):
        self.payload, self.stop_reason, self.request = payload, stop_reason, None
        self.beta = self
        self.messages = self

    def create(self, **kwargs):
        self.request = kwargs
        response = type("Response", (), {})()
        response.stop_reason = self.stop_reason
        response.stop_details = None
        response.content = [_StubBlock(json.dumps(self.payload))]
        return response


def _raw_atom(**fields):
    atom = {"text": "x", "condition_type": "pre_condition", **{f: [] for f in LIST_FIELDS}}
    atom.update(fields)
    return atom


class _StubEncoder:
    """Maps strings to fixed unit vectors so similarity is known in advance."""

    VECTORS = {"care provider": [1.0, 0.0], "doctor": [0.96, 0.28], "patient": [0.0, 1.0]}

    def encode(self, texts, normalize_embeddings=True, convert_to_numpy=True):
        import numpy as np
        return np.array([self.VECTORS.get(t, [0.0, 0.0]) for t in texts])


class SemanticExtensionTest(unittest.TestCase):
    def test_unseen_query_tag_gets_similar_corpus_tags(self):
        from src.data.query import embed_vocabulary, extend_semantic_lookup
        vocab = {"actors": ["care provider", "patient"]}
        emb = embed_vocabulary(vocab, _StubEncoder())
        base = {("actors", "patient"): []}
        query = [{"atom_id": "Q.1", "actors": ["doctor", "patient"]}]
        lookup = extend_semantic_lookup(base, query, emb, _StubEncoder(), threshold=0.55)
        self.assertEqual([t for t, _ in lookup[("actors", "doctor")]], ["care provider"])
        self.assertNotIn(("actors", "doctor"), base)  # original left untouched


class QueryTest(unittest.TestCase):
    """Natural-language query decomposition, without calling the API."""

    @classmethod
    def setUpClass(cls):
        cls.df = only_synthetic(build_atom_table(ANNOTATIONS)).reset_index(drop=True)
        cls.vocab = unique_tags_by_field(cls.df)

    def test_parse_snaps_to_vocabulary_and_dedupes(self):
        actor = self.vocab["actors"][0]
        payload = {"atoms": [_raw_atom(actors=[actor.upper(), f"  {actor} ", "brand new actor"])]}
        atoms = parse_query_atoms(payload, self.vocab)
        self.assertEqual(atoms[0]["actors"], [actor, "brand new actor"])
        self.assertEqual(atoms[0]["atom_id"], "Q.1")

    def test_text_to_query_atoms_sends_vocabulary_and_schema(self):
        client = _StubClient({"atoms": [_raw_atom(acts=["requests"])]})
        atoms = text_to_query_atoms("Can I ask for deletion?", self.vocab, client=client)
        self.assertEqual(atoms[0]["acts"], ["requests"])
        system_text = client.request["system"][0]["text"]
        self.assertIn(self.vocab["actors"][0], system_text)
        self.assertEqual(client.request["output_config"]["format"]["type"], "json_schema")
        self.assertEqual(client.request["messages"][0]["content"], "Can I ask for deletion?")

    def test_refusal_raises(self):
        client = _StubClient({"atoms": []}, stop_reason="refusal")
        with self.assertRaises(RuntimeError):
            text_to_query_atoms("x", self.vocab, client=client)

    def test_retrieve_for_text_ranks_the_source_article_first(self):
        source = self.df.iloc[0]
        raw = _raw_atom(**{f: list(source[f]) for f in LIST_FIELDS})
        result = retrieve_for_text("anything", self.df, client=_StubClient({"atoms": [raw]}))
        top = result["article_ranking"].iloc[0]
        self.assertEqual(top["article_id"], source["article_id"])
        self.assertEqual(top["n_query_atoms_matched"], 1)

    def test_coverage_is_average_over_query_atoms(self):
        source = self.df.iloc[0]
        raw = _raw_atom(**{f: list(source[f]) for f in LIST_FIELDS})
        unrelated = _raw_atom(actors=["nobody"], residual=["nothing at all"])
        client = _StubClient({"atoms": [raw, unrelated]})
        ranking = retrieve_for_text("anything", self.df, client=client)["article_ranking"]
        for r in ranking.itertuples(index=False):
            self.assertAlmostEqual(r.coverage, r.score / 2, places=3)
        self.assertLessEqual(ranking["coverage"].max(), 1.0001)


_BWB_FIXTURE = b"""<?xml version="1.0" encoding="utf-8"?>
<toestand><wetgeving><wet-besluit><wettekst>
<boek bwb-ng-variabel-deel="/Boek7" label="Boek 7"><kop><label>Boek</label><nr>7</nr><titel>Bijzondere overeenkomsten</titel></kop>
<titeldeel bwb-ng-variabel-deel="/Boek7/Titeldeel7" label="Titel 7"><kop><label>Titel</label><nr>7</nr><titel>Opdracht</titel></kop>
<afdeling bwb-ng-variabel-deel="/Boek7/Titeldeel7/Afdeling5" label="Afdeling 5"><kop><label>Afdeling</label><nr>5</nr><titel>WGBO</titel></kop>
  <artikel bwb-ng-variabel-deel="/Boek7/Titeldeel7/Afdeling5/Artikel454" label="Artikel 454" inwerking="2020-01-01" status="goed">
    <kop><label>Artikel</label><nr>454</nr></kop>
    <lid><lidnr>1</lidnr><al>De hulpverlener richt een dossier in.</al><meta-data><x>noise</x></meta-data></lid>
    <lid><lidnr>3</lidnr><al>Onverminderd <intref bwb-id="BWBR0005290" bwb-ng-variabel-deel="/Boek7/Titeldeel7/Afdeling5/Artikel455/Lid2">artikel 455</intref> bewaart hij het dossier.</al></lid>
  </artikel>
  <artikel bwb-ng-variabel-deel="/Boek7/Titeldeel7/Afdeling5/Artikel449" label="Artikel 449" status="goed">
    <kop><label>Artikel</label><nr>449</nr></kop>
    <al>Een enkele alinea:</al><lijst><li><li.nr>a.</li.nr><al>eerste;</al></li></lijst>
  </artikel>
  <artikel bwb-ng-variabel-deel="/Boek7/Titeldeel7/Afdeling5/Artikel458" label="Artikel 458" status="vervallen">
    <kop><label>Artikel</label><nr>458</nr></kop><al>Vervallen.</al>
  </artikel>
</afdeling></titeldeel></boek>
</wettekst></wet-besluit></wetgeving></toestand>"""


class StatuteParsingTest(unittest.TestCase):
    """Parsing of official BWB statute XML (src/data/statutes.py)."""

    @classmethod
    def setUpClass(cls):
        from src.data.statutes import parse_bwb
        cls.articles, cls.repealed = parse_bwb(
            _BWB_FIXTURE, ["/Boek7/Titeldeel7/Afdeling5"], bwb_id="BWBR0005290", version="v1")
        cls.by_id = {a["article_id"]: a for a in cls.articles}

    def test_articles_and_repealed(self):
        self.assertEqual(sorted(self.by_id), ["7:449", "7:454"])
        self.assertEqual(self.repealed, ["7:458"])

    def test_tree_and_metadata(self):
        a = self.by_id["7:454"]
        self.assertEqual((a["boek"], a["titeldeel"], a["afdeling"]), ("Boek 7", "Titel 7", "Afdeling 5"))
        self.assertEqual((a["titeldeel_name"], a["afdeling_name"]), ("Opdracht", "WGBO"))
        self.assertEqual(a["in_force_since"], "2020-01-01")

    def test_lid_text_excludes_numbers_and_metadata(self):
        lids = self.by_id["7:454"]["lids"]
        self.assertEqual([l["lid"] for l in lids], ["1", "3"])
        self.assertEqual(lids[0]["text"], "De hulpverlener richt een dossier in.")
        self.assertNotIn("noise", self.by_id["7:454"]["text"])

    def test_unnumbered_article_gets_single_lid_with_list(self):
        lids = self.by_id["7:449"]["lids"]
        self.assertEqual(len(lids), 1)
        self.assertEqual(lids[0]["lid"], "")
        self.assertEqual(lids[0]["text"], "Een enkele alinea: a. eerste;")

    def test_reference_resolved_to_article_and_lid(self):
        (ref,) = self.by_id["7:454"]["references"]
        self.assertEqual((ref["source_lid"], ref["target_article"], ref["target_lid"]), ("3", "7:455", "2"))
        self.assertEqual(ref["ref_text"], "artikel 455")


_JUDGMENT_XML = """<?xml version="1.0" encoding="utf-8"?>
<open-rechtspraak>
  <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" xmlns:dcterms="http://purl.org/dc/terms/"
           xmlns:ecli="https://e-justice.europa.eu/ecli" xmlns:psi="http://psi.rechtspraak.nl/">
    <rdf:Description>
      <dcterms:identifier>ECLI:NL:GHXX:2020:1</dcterms:identifier>
      <dcterms:creator>Gerechtshof X</dcterms:creator>
      <dcterms:date>2020-05-01</dcterms:date>
      <psi:procedure>Hoger beroep</psi:procedure>
      <dcterms:relation ecli:resourceIdentifier="ECLI:NL:RBXX:2018:9">Eerste aanleg</dcterms:relation>
    </rdf:Description>
  </rdf:RDF>
  <inhoudsindicatie xmlns="http://www.rechtspraak.nl/schema/rechtspraak-1.0"><para>Aansprakelijkheid huisarts, art. 7:453 BW.</para></inhoudsindicatie>
  <uitspraak xmlns="http://www.rechtspraak.nl/schema/rechtspraak-1.0">
    <section><title><nr>1</nr> Het geding</title><para>Procedure.</para></section>
    <section><title><nr>2</nr> De feiten</title>
      <para>2.1 De huisarts heeft om 7:45 uur de patiënt gezien.</para>
      <para>2.2 Eiser beroept zich op artikel 7:453 BW.</para></section>
    <section><title><nr>3</nr> De beoordeling</title><para>Ook 6:162 BW is van toepassing.</para></section>
  </uitspraak>
</open-rechtspraak>"""


class CaselawTest(unittest.TestCase):
    """Judgment parsing and citation masking for study A (src/data/caselaw.py)."""

    def test_parse_metadata_relations_and_facts_section(self):
        from src.data.caselaw import parse_judgment
        j = parse_judgment(_JUDGMENT_XML.encode("utf-8"))
        self.assertEqual((j["ecli"], j["court"], j["procedure"]),
                         ("ECLI:NL:GHXX:2020:1", "Gerechtshof X", "Hoger beroep"))
        self.assertEqual(j["related"], ["ECLI:NL:RBXX:2018:9"])
        self.assertEqual(j["facts_method"], "section")
        self.assertTrue(j["facts"].startswith("2.1 De huisarts"))
        self.assertNotIn("6:162", j["facts"])  # the next section is not part of the facts

    def test_facts_from_numbered_heading_without_sections(self):
        from src.data.caselaw import _facts_from_text
        text = "1. Het geding Procedure. 2. De feiten 2.1 Eiser is geopereerd. 3. De beoordeling Iets."
        self.assertEqual(_facts_from_text(text), "2.1 Eiser is geopereerd.")

    def test_citations_ignore_other_laws(self):
        from src.data.caselaw import bw_citations
        self.assertEqual(bw_citations("artikel 7:453 BW en artikel 7:11 Awb"), {"7:453"})

    def test_masking_removes_every_corpus_reference(self):
        from src.data.caselaw import bw_citations, mask_citations
        cases = {
            "ingevolge art. 7:453 BW van een goed hulpverlener": "ingevolge [ARTIKEL] van een goed hulpverlener",
            "de artikelen 7:446 en 7:453 BW en artikel 6:162 lid 1 BW": "de [ARTIKEL] en [ARTIKEL]",
            "artikel 453 van Boek 7 BW": "[ARTIKEL]",
            "(7:446 e.v. BW) en de WGBO": "([ARTIKEL]) en de [WET]",
            "artt. 7:448 lid 2, sub b en 7:450 BW": "[ARTIKEL]",
        }
        for text, expected in cases.items():
            masked, n = mask_citations(text)
            self.assertEqual(masked, expected)
            self.assertGreater(n, 0)
            self.assertEqual(bw_citations(masked), set())

    def test_masking_keeps_clock_times(self):
        from src.data.caselaw import mask_citations
        self.assertEqual(mask_citations("om 7:45 uur en om 19:20 uur"), ("om 7:45 uur en om 19:20 uur", 0))


class RetrievalMetricsTest(unittest.TestCase):
    """Baseline retrieval and multi-answer metrics (src/similarity/retrieval.py)."""

    def test_rank_metrics_with_two_answers(self):
        from src.similarity.retrieval import rank_metrics
        m = rank_metrics(["a", "b", "c", "d"], {"b", "d"}, ks=(1, 2, 4), ndcg_k=4)
        self.assertEqual((m["first_rank"], m["mrr"]), (2, 0.5))
        self.assertEqual((m["recall@1"], m["recall@2"], m["recall@4"]), (0.0, 0.5, 1.0))
        self.assertTrue(0 < m["ndcg@4"] < 1)
        self.assertEqual(rank_metrics(["b", "d", "a"], {"b", "d"}, ndcg_k=4)["ndcg@4"], 1.0)

    def test_ties_are_averaged_not_ordered(self):
        import numpy as np
        from src.similarity.retrieval import evaluate
        # All four articles tied: expected recall@1 of one answer is 1/4
        _, summary = evaluate(np.zeros((1, 4)), ["a", "b", "c", "d"], [["c"]], tie_draws=400)
        self.assertAlmostEqual(summary["recall@1"], 0.25, delta=0.06)

    def test_bm25_ranks_the_matching_article_first(self):
        import numpy as np
        from src.similarity.retrieval import bm25_scores
        docs = ["De hulpverlener houdt een dossier bij.", "De verhuurder onderhoudt de woning.",
                "Schade wordt vergoed."]
        scores = bm25_scores(["het ziekenhuis is het dossier van de patiënt kwijt"], docs)
        self.assertEqual(int(np.argmax(scores[0])), 0)

    def test_fusion_rewards_agreement_and_ignores_scale(self):
        import numpy as np
        from src.similarity.retrieval import rrf_fusion, zscore_fusion
        # Scales differ wildly. Both systems rank article 0 first; they disagree on 1 and 2
        a = np.array([[100.0, 95.0, 0.0]])
        b = np.array([[0.9, 0.1, 0.5]])
        for fused in (zscore_fusion(a, b), rrf_fusion(a, b)):
            self.assertEqual(int(np.argmax(fused[0])), 0)
        # Scaling one system does not change the RRF ranking
        np.testing.assert_array_equal(rrf_fusion(a, b), rrf_fusion(a * 1000, b))


if __name__ == "__main__":
    unittest.main()
