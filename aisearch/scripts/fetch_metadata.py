#! /usr/bin/env python3
"""Fetch RDF triples grouped by subject and convert to natural language sentences.

This script fetches data from GeneNetwork's SPARQL endpoint, grouping triples by
subject and converting them to human-readable sentences. It processes different
entity types in parallel using threads and writes each type to its own file.
"""

import json
import os
import time
import random
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple, Optional
from SPARQLWrapper import SPARQLWrapper, JSON
from urllib.error import HTTPError

SPARQL_ENDPOINT = "https://rdf.genenetwork.org/sparql"
DEFAULT_GRAPH = "http://rdf.genenetwork.org/v1"

PREFIXES = {
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
    "skos": "http://www.w3.org/2004/02/skos/core#",
    "dct": "http://purl.org/dc/terms/",
    "gnc": "http://rdf.genenetwork.org/v1/category/",
    "gnt": "http://rdf.genenetwork.org/v1/term/",
    "gn": "http://rdf.genenetwork.org/v1/id/",
    "xkos": "http://rdf-vocabulary.ddialliance.org/xkos#",
    "fabio": "http://purl.org/spar/fabio/",
    "prism": "http://prismstandard.org/namespaces/basic/2.0/",
    "owl": "http://www.w3.org/2002/07/owl#",
    "xsd": "http://www.w3.org/2001/XMLSchema#",
    "dcat": "http://www.w3.org/ns/dcat#",
    "qb": "http://purl.org/linked-data/cube#",
    "schema": "https://schema.org/",
    "pubmed": "http://rdf.ncbi.nlm.nih.gov/pubmed/",
    "obo": "http://purl.obolibrary.org/obo/",
    "bfo": "http://purl.obolibrary.org/obo/BFO_",
    "sdmx-measure": "http://purl.org/linked-data/sdmx/2009/measure#",
    "foaf": "http://xmlns.com/foaf/0.1/",
}

TEMPLATES = {
    # RDF / RDFS
    "http://www.w3.org/1999/02/22-rdf-syntax-ns#type": "is a {obj}",
    "http://www.w3.org/2000/01/rdf-schema#label": "is labeled as '{obj}'",
    "http://www.w3.org/2000/01/rdf-schema#comment": "has the note: {obj}",
    # SKOS
    "http://www.w3.org/2004/02/skos/core#prefLabel": "is called '{obj}'",
    "http://www.w3.org/2004/02/skos/core#altLabel": "is also known as '{obj}'",
    "http://www.w3.org/2004/02/skos/core#definition": "is defined as: {obj}",
    "http://www.w3.org/2004/02/skos/core#member": "includes member {obj}",
    "http://www.w3.org/2004/02/skos/core#notation": "has notation {obj}",
    "http://www.w3.org/2004/02/skos/core#inScheme": "is in scheme {obj}",
    # DCTERMS
    "http://purl.org/dc/terms/title": "has title '{obj}'",
    "http://purl.org/dc/terms/description": "is described as: '{obj}'",
    "http://purl.org/dc/terms/identifier": "has identifier {obj}",
    "http://purl.org/dc/terms/abstract": "has abstract '{obj}'",
    "http://purl.org/dc/terms/creator": "was created by {obj}",
    "http://purl.org/dc/terms/contributor": "has contributor {obj}",
    "http://purl.org/dc/terms/created": "was created on {obj}",
    "http://purl.org/dc/terms/references": "references {obj}",
    "http://purl.org/dc/terms/hasVersion": "has version {obj}",
    # GeneNetwork Terms
    "http://rdf.genenetwork.org/v1/term/has_set_code": "provides a unique identifier code for a resource set {obj}",
    "http://rdf.genenetwork.org/v1/term/genetic_type": "describes the genetic architecture of a resource set as {obj}",
    "http://rdf.genenetwork.org/v1/term/has_species": "belongs to species {obj}",
    "http://rdf.genenetwork.org/v1/term/has_strain": "lists all strains that belong to this resource including {obj}",
    "http://rdf.genenetwork.org/v1/term/has_taxonomic_family": "links to taxonomic family {obj}",
    "http://rdf.genenetwork.org/v1/term/has_uniprot_taxon_id": "has UniProt taxonomic id {obj}",
    "http://rdf.genenetwork.org/v1/term/short_name": "has short name {obj}",
    "http://rdf.genenetwork.org/v1/term/uses_mapping_method": "uses mapping method {obj}",
    "http://rdf.genenetwork.org/v1/term/has_reference_population": "has reference population {obj}",
    "http://rdf.genenetwork.org/v1/term/alias": "has alias {obj}",
    "http://rdf.genenetwork.org/v1/term/has_phenotype": "measures phenotype {obj}",
    "http://rdf.genenetwork.org/v1/term/abbreviation": "has abbreviation {obj}",
    "http://rdf.genenetwork.org/v1/term/has_lab_code": "has lab code {obj}",
    "http://rdf.genenetwork.org/v1/term/lod_score": "has LOD score (Peak -logP) {obj}",
    "http://rdf.genenetwork.org/v1/term/mean": "has mean value {obj}",
    "http://rdf.genenetwork.org/v1/term/submitter": "was submitted by {obj}",
    "http://rdf.genenetwork.org/v1/term/has_genotype_files": "has genotype files {obj}",
    "http://rdf.genenetwork.org/v1/term/has_marker_count": "has {obj} DNA markers/SNPs",
    "http://rdf.genenetwork.org/v1/term/symbol": "has gene symbol {obj}",
    "http://rdf.genenetwork.org/v1/term/chr": "is on chromosome {obj}",
    "http://rdf.genenetwork.org/v1/term/chromosome": "is on chromosome {obj}",
    "http://rdf.genenetwork.org/v1/term/mb": "is at {obj} megabases",
    "http://rdf.genenetwork.org/v1/term/sequence": "has sequence {obj}",
    "http://rdf.genenetwork.org/v1/term/has_target_id": "has target id {obj}",
    "http://rdf.genenetwork.org/v1/term/targets_region": "targets region {obj}",
    "http://rdf.genenetwork.org/v1/term/has_specificity": "has specificity {obj}",
    "http://rdf.genenetwork.org/v1/term/has_blat_score": "has BLAT score {obj}",
    "http://rdf.genenetwork.org/v1/term/has_blat_mb_start": "has BLAT start position {obj}",
    "http://rdf.genenetwork.org/v1/term/has_blat_mb_end": "has BLAT end position {obj}",
    "http://rdf.genenetwork.org/v1/term/has_blat_seq": "has BLAT sequence {obj}",
    "http://rdf.genenetwork.org/v1/term/has_target_seq": "has target sequence {obj}",
    "http://rdf.genenetwork.org/v1/term/has_homologene_id": "has HomoloGene ID {obj}",
    "http://rdf.genenetwork.org/v1/term/has_uniprot_id": "has UniProt ID {obj}",
    "http://rdf.genenetwork.org/v1/term/has_pub_chem_id": "has PubChem ID {obj}",
    "http://rdf.genenetwork.org/v1/term/has_kegg_id": "has KEGG ID {obj}",
    "http://rdf.genenetwork.org/v1/term/has_omim_id": "has OMIM ID {obj}",
    "http://rdf.genenetwork.org/v1/term/has_chebi_id": "has ChEBI ID {obj}",
    "http://rdf.genenetwork.org/v1/term/initial": "has initial {obj}",
    "http://rdf.genenetwork.org/v1/term/reason": "was modified because {obj}",
    "http://rdf.genenetwork.org/v1/term/belongs_to_category": "belongs to category {obj}",
    "http://rdf.genenetwork.org/v1/term/has_gene_id": "has gene ID {obj}",
    "http://rdf.genenetwork.org/v1/term/gene_symbol": "has gene symbol {obj}",
    "http://rdf.genenetwork.org/v1/term/transcript": "has transcript {obj}",
    "http://rdf.genenetwork.org/v1/term/strand": "is on strand {obj}",
    "http://rdf.genenetwork.org/v1/term/tx_start": "has transcription start {obj}",
    "http://rdf.genenetwork.org/v1/term/tx_end": "has transcription end {obj}",
    "http://rdf.genenetwork.org/v1/term/has_align_id": "has alignment ID {obj}",
    "http://rdf.genenetwork.org/v1/term/has_protein_id": "has protein ID {obj}",
    "http://rdf.genenetwork.org/v1/term/has_rgd_id": "has RGD ID {obj}",
    "http://rdf.genenetwork.org/v1/term/has_geo_series_id": "has GEO series ID {obj}",
    "http://rdf.genenetwork.org/v1/term/has_go_tree_value": "has GO tree value {obj}",
    "http://rdf.genenetwork.org/v1/term/has_case_info": "has case info: '{obj}'",
    "http://rdf.genenetwork.org/v1/term/has_citation": "has citation: '{obj}'",
    "http://rdf.genenetwork.org/v1/term/has_contributors": "has contributors: '{obj}'",
    "http://rdf.genenetwork.org/v1/term/has_data_processing_info": "has data processing info: '{obj}'",
    "http://rdf.genenetwork.org/v1/term/has_experiment_design": "has experiment design: '{obj}'",
    "http://rdf.genenetwork.org/v1/term/has_experiment_design_info": "has experiment design info: '{obj}'",
    "http://rdf.genenetwork.org/v1/term/has_experiment_type": "has experiment type: '{obj}'",
    "http://rdf.genenetwork.org/v1/term/has_molecular_trait": "has molecular trait {obj}",
    "http://rdf.genenetwork.org/v1/term/has_phenotype_data": "has phenotype data {obj}",
    "http://rdf.genenetwork.org/v1/term/has_phenotype_trait": "has phenotype trait {obj}",
    "http://rdf.genenetwork.org/v1/term/has_platform_info": "has platform info: {obj}",
    "http://rdf.genenetwork.org/v1/term/has_probeset_data": "has probeset data {obj}",
    "http://rdf.genenetwork.org/v1/term/has_samples": "has samples {obj}",
    "http://rdf.genenetwork.org/v1/term/has_specifics": "has specifics: '{obj}'",
    "http://rdf.genenetwork.org/v1/term/has_summary": "has summary: '{obj}'",
    "http://rdf.genenetwork.org/v1/term/has_tissue_info": "has tissue info: '{obj}'",
    "http://rdf.genenetwork.org/v1/term/uses_genechip": "uses genechip platform {obj}",
    "http://rdf.genenetwork.org/v1/term/uses_normalization_method": "uses normalization method {obj}",
    "http://rdf.genenetwork.org/v1/term/has_probeset": "has probeset {obj}",
    "http://rdf.genenetwork.org/v1/term/has_genotype_data": "has genotype data {obj}",
    "http://rdf.genenetwork.org/v1/term/has_sequence": "has sequence {obj}",
    "http://rdf.genenetwork.org/v1/term/source": "has source {obj}",
    "http://rdf.genenetwork.org/v1/term/has_kg_id": "has KG ID {obj}",
    "http://rdf.genenetwork.org/v1/term/has_unigen_id": "has UniGene ID {obj}",
    "http://rdf.genenetwork.org/v1/term/additive": "has additive effect {obj}",
    "http://rdf.genenetwork.org/v1/term/se": "has standard error {obj}",
    "http://rdf.genenetwork.org/v1/term/pvalue": "has p-value {obj}",
    "http://rdf.genenetwork.org/v1/term/h2": "has heritability (h2) {obj}",
    "http://rdf.genenetwork.org/v1/term/assigned_species": "has assigned species {obj}",
    "http://rdf.genenetwork.org/v1/term/has_family_order_id": "has family order ID {obj}",
    "http://rdf.genenetwork.org/v1/term/has_population_order_id": "has population order ID {obj}",
    "http://rdf.genenetwork.org/v1/term/locus": "has locus {obj}",
    "http://rdf.genenetwork.org/v1/term/mb_mm8": "has position (mm8) {obj} megabases",
    # FOAF
    "http://xmlns.com/foaf/0.1/homepage": "has homepage {obj}",
    "http://xmlns.com/foaf/0.1/name": "has name '{obj}'",
    "http://xmlns.com/foaf/0.1/familyName": "has family name '{obj}'",
    "http://xmlns.com/foaf/0.1/givenName": "has given name '{obj}'",
    "http://xmlns.com/foaf/0.1/mbox": "has email {obj}",
    # XKOS
    "http://rdf-vocabulary.ddialliance.org/xkos#depth": "is at depth level {obj}",
    # FABIO
    "http://purl.org/spar/fabio/hasPubMedId": "has PubMed ID {obj}",
    "http://purl.org/spar/fabio/Journal": "was published in journal '{obj}'",
    "http://purl.org/spar/fabio/page": "is on pages {obj}",
    "http://purl.org/spar/fabio/hasPublicationYear": "was published in year {obj}",
    # PRISM
    "http://prismstandard.org/namespaces/basic/2.0/volume": "is in volume {obj}",
    "http://prismstandard.org/namespaces/basic/2.0/publicationDate": "was published on {obj}",
    # DCAT
    "http://www.w3.org/ns/dcat#distribution": "has distribution {obj}",
    "http://www.w3.org/ns/dcat#isPartOf": "is part of {obj}",
    # OWL
    "http://www.w3.org/2002/07/owl#equivalentClass": "is equivalent to {obj}",
}


TYPE_QUERIES = [
    "a dcat:Dataset",
    "a gnc:set",
    "a gnc:species",
    "a gnc:reference_population",
    "a skos:Concept",
    "a gnc:molecular_trait",
    "a gnc:phenotype",
    "a gnc:strain",
    "a fabio:ResearchPaper",
    "a gnc:gene",
    "a gnc:gene_symbol",
    "a gnc:dna_marker",
    "a gnc:ncbi_wiki_entry",
]


def type_pattern_to_filename(type_pattern: str) -> str:
    """Convert a type pattern to a safe filename."""
    return f"{type_pattern.split(':')[-1].lower()}.json"


def uri_to_prefixed(uri: str) -> str:
    """Convert full URI to prefixed form."""
    for prefix, ns in PREFIXES.items():
        if uri.startswith(ns):
            return f"{prefix}:{uri[len(ns):]}"
    return uri


def triple_to_sentence(subj: str, pred: str, obj: str) -> str:
    """Convert triple to natural language sentence."""
    obj_prefixed = uri_to_prefixed(obj)
    if pred in TEMPLATES:
        return f"{subj} {TEMPLATES[pred].format(obj=obj_prefixed)}."
    pred_prefixed = uri_to_prefixed(pred)
    return f"{subj} {pred_prefixed} {obj_prefixed}."


def fetch_with_retry(
    sparql: SPARQLWrapper,
    max_retries: int = 3,
    base_delay: float = 2.0
) -> Optional[Dict]:
    """Execute SPARQL query with retry logic and exponential backoff."""
    for attempt in range(max_retries):
        try:
            return sparql.queryAndConvert()
        except HTTPError as e:
            if e.code == 504 and attempt < max_retries - 1:
                # Exponential backoff with jitter
                delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                print(f"    Timeout (attempt {attempt + 1}/{max_retries}), retrying in {delay:.1f}s...")
                time.sleep(delay)
            else:
                raise
    return None


def fetch_count_for_type(type_pattern: str) -> int:
    """Get the total count of triples for a type pattern.

    Args:
        type_pattern: The type pattern to count

    Returns:
        Total count of triples
    """
    sparql = SPARQLWrapper(SPARQL_ENDPOINT)
    sparql.setReturnFormat(JSON)
    sparql.setTimeout(60)

    query = f"""
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX dcat: <http://www.w3.org/ns/dcat#>
    PREFIX gnc: <http://rdf.genenetwork.org/v1/category/>
    PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
    PREFIX fabio: <http://purl.org/spar/fabio/>
    PREFIX foaf: <http://xmlns.com/foaf/0.1/>

    SELECT (COUNT(*) AS ?count)
    FROM <{DEFAULT_GRAPH}>
    WHERE {{
      ?s ?p ?o ;
         {type_pattern} .
    }}"""

    sparql.setQuery(query)

    try:
        result = fetch_with_retry(sparql)
        if result:
            count_str = result["results"]["bindings"][0]["count"]["value"]
            return int(count_str)
    except Exception as e:
        print(f"    Warning: Could not get count: {e}")

    return 0


def fetch_by_type_pattern(
    type_pattern: str,
    page_size: int = 5000
) -> List[Dict]:
    """Fetch triples for a specific type pattern using pagination.

    First gets the count, then calculates pages needed.

    Args:
        type_pattern: The type pattern to query
        page_size: Number of triples per page
    Returns:

        List of SPARQL binding dictionaries
    """
    # First, get the count
    print(f"  Getting count for {type_pattern}...")
    total_count = fetch_count_for_type(type_pattern)

    if total_count == 0:
        print(f"  No data found for {type_pattern}")
        return []

    num_pages = (total_count + page_size - 1) // page_size  # Ceiling division
    print(f"  Total: {total_count} triples, fetching {num_pages} pages...")

    all_bindings = []
    sparql = SPARQLWrapper(SPARQL_ENDPOINT)
    sparql.setReturnFormat(JSON)
    sparql.setTimeout(120000)

    for page in range(num_pages):
        offset = page * page_size

        query = f"""
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX dcat: <http://www.w3.org/ns/dcat#>
        PREFIX gnc: <http://rdf.genenetwork.org/v1/category/>
        PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
        PREFIX fabio: <http://purl.org/spar/fabio/>
        PREFIX foaf: <http://xmlns.com/foaf/0.1/>

        SELECT ?s ?p ?o
        FROM <{DEFAULT_GRAPH}>
        WHERE {{
          ?s ?p ?o ;
             {type_pattern} .
        }}
        LIMIT {page_size}
        OFFSET {offset}"""

        sparql.setQuery(query)

        try:
            result = fetch_with_retry(sparql)
            if result is None:
                break

            bindings = result.get("results", {}).get("bindings", [])
            all_bindings.extend(bindings)

            # Progress indicator
            if (page + 1) % 5 == 0 or page == num_pages - 1:
                print(f"    Fetched page {page + 1}/{num_pages} ({len(all_bindings)}/{total_count} triples) :: {type_pattern_to_filename(type_pattern)}")

            # Small delay between pages
            if page < num_pages - 1:
                time.sleep(0.3)

        except Exception as e:
            print(f"    ERROR on page {page + 1}: {e}")
            break

    return all_bindings


def group_by_subject(bindings: List[Dict]) -> Dict[str, List[Tuple[str, str]]]:
    """Group triples by subject URI."""
    grouped = defaultdict(list)
    for b in bindings:
        subj = b["s"]["value"]
        pred = b["p"]["value"]
        obj = b["o"]["value"]
        grouped[subj].append((pred, obj))
    return grouped


def build_sentences(grouped: Dict[str, List[Tuple[str, str]]]) -> List[str]:
    """Build list of sentence groups, one per subject."""
    result = []
    for subj_uri, triples in grouped.items():
        subject = uri_to_prefixed(subj_uri)
        sentences = [triple_to_sentence(subject, pred, obj) for pred, obj in triples]
        result.append(" ".join(sentences))
    return result


def process_type_query(
    type_pattern: str,
    output_dir: str,
    page_size: int = 5000
) -> Tuple[str, int, bool]:
    """Process a single type query and write results to a file.

    Args:
        type_pattern: The type pattern to query
        output_dir: Directory to write the output file
        page_size: Number of triples per page

    Returns:
        Tuple of (type_pattern, count, success)
    """
    filename = type_pattern_to_filename(type_pattern)
    output_path = os.path.join(output_dir, filename)

    # Add delay at start to stagger requests when running in parallel
    time.sleep(random.uniform(0.5, 2.0))

    try:
        print(f"Fetching {type_pattern}...")
        bindings = fetch_by_type_pattern(type_pattern, page_size=page_size)

        if not bindings:
            with open(output_path, "w") as f:
                json.dump([], f, indent=2)
            return type_pattern, 0, True

        print(f"  Processing {len(bindings)} triples for {type_pattern}")

        grouped = group_by_subject(bindings)
        print(f"  Grouped into {len(grouped)} subjects")

        sentences = build_sentences(grouped)

        with open(output_path, "w") as f:
            json.dump(sentences, f, indent=2)

        print(f"  Wrote {len(sentences)} sentences to {filename}")
        return type_pattern, len(sentences), True

    except Exception as e:
        print(f"  ERROR processing {type_pattern}: {e}")
        return type_pattern, 0, False


def main():
    parser = argparse.ArgumentParser(
        description="Fetch RDF triples as natural language, one file per type"
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="metadata_output",
        help="Output directory for JSON files (default: metadata_output)"
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=2,
        help="Number of parallel workers (default: 2, use 1 for sequential)"
    )
    parser.add_argument(
        "--page-size", "-p",
        type=int,
        default=5000,
        help="Number of triples per page (default: 5000)"
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory: {args.output_dir}")
    print(f"Processing {len(TYPE_QUERIES)} type queries with {args.workers} workers...")
    print(f"Page size: {args.page_size}")
    print()

    results = []

    if args.workers == 1:
        # Sequential processing - more polite to the server
        for type_pattern in TYPE_QUERIES:
            result = process_type_query(type_pattern, args.output_dir, args.page_size)
            results.append(result)
            # Delay between types
            time.sleep(1.0)
    else:
        # Parallel processing with limited workers
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_to_type = {
                executor.submit(process_type_query, tq, args.output_dir, args.page_size): tq
                for tq in TYPE_QUERIES
            }

            for future in as_completed(future_to_type):
                type_pattern = future_to_type[future]
                try:
                    _, count, success = future.result()
                    results.append((type_pattern, count, success))
                except Exception as e:
                    print(f"  UNEXPECTED ERROR for {type_pattern}: {e}")
                    results.append((type_pattern, 0, False))

    # Summary
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)

    successful = sum(1 for _, _, s in results if s)
    total_sentences = sum(c for _, c, s in results if s)

    for type_pattern, count, success in sorted(results):
        status = "✓" if success else "✗"
        print(f"{status} {type_pattern}: {count} sentences")

    print()
    print(f"Successfully processed: {successful}/{len(TYPE_QUERIES)}")
    print(f"Total sentences written: {total_sentences}")
    print(f"Output files in: {args.output_dir}")


if __name__ == "__main__":
    print("script starting", flush=True)
    main()
