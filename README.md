<div align="center">
  <h1>**GNAIS**</h1>
</div>

## What is GNAIS?

**GNAIS** (GeneNetwork AI Search) is a search tool that helps digest metadata around GeneNetwork using language models. It enriches context of LLMs using GeneNetwork RDF data and metadata.

**GNAIS** can perform:
- document RAG search
- graph RAG search
- agentic search
- hybrid search

**GNAIS** was implemented using [DSPy](https://dspy.ai/).  Switching between LLM providers for the text generation model is as easy as changing a variable declaration.

## Demos

Here are a few examples of response generated with GNAIS with claude-haiku-4-5-20251001.

1. Digest and explain phenotype datasets around cognition measured in BXD mice*

Response:
```
The BXD mouse phenotype datasets contain several cognition-related measurements.  The primary focus is on working memory deficits, with three related traits measuring age at onset (AAO) of working memory impairment in non-transgenic BXD littermates from the AD-BXD population: a combined sex-averaged trait (mean 11.12 months, LOD 1.89), a male-specific trait (mean 10.56 months, LOD 4.57), and a female-specific trait (mean 11.70 months, LOD 2.06).  These traits are part of a larger study examining cognitive aging in transgenic and non-transgenic AD-BXD mice.  Additionally, the dataset includes a motor impulsivity trait measured via the 5-choice serial reaction time task (5CSRT) in males, which assesses premature responding behavior (mean 9.35, LOD 3.60).  All cognition traits were submitted by Sarah Neuner and Catherine Kaczorowski, and the working memory traits are referenced in unpublished studies, while the impulsivity trait is published in PubMed:24703509."
```

2. Compare traits related to the central nervous system measured in BXD, BDL, CCB and MDP mice

Response:
```
Central nervous system traits across the four mouse datasets show distinct research focuses: BXD mice (10 traits) include stress-induced changes (CMS treatment effects on organ weights and sacrifice age), morphological measurements (brain region weights), and locomotor behavior in novel environments.  BDL mice (1 trait) measure aging-related fear conditioning and baseline anxiety responses.  CCB mice (8 traits) emphasize behavioral and cognitive phenotypes including fear learning/extinction, anxiety-like behavior, and social interaction measures (dominance, aggression, sniffing).  MDP mice (1 trait) focus on pharmacological responses, specifically ethanol sensitization.  The BXD dataset provides the most comprehensive CNS phenotyping with 10 traits covering pharmacology, morphology, and behavior.  CCB traits are particularly rich in behavioral/cognitive measures, while BDL and MDP datasets each contribute specialized phenotypes for aging and drug response respectively.  All traits are measured in adult or aging mice, with BXD and CCB using controlled behavioral testing paradigms and BXD additionally measuring stress-induced physiological changes.
```

## Installing GNAIS

**GNAIS** was packaged.  You can install it in your virtual environment using the following commands:

```python
git clone https://github.com/genenetwork/gn-ai.git
cd aisearch
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Using GNAIS

To run **GNAIS**, you need to define a few parameters in your bash environment. We recommend setting them in a file.  Check `aisearch/.env.example`. Once defined, you can run your search query using scripts in `aisearch/scripts`

```python
python aisearch/scripts/grag_search.py <your-query>
```


## Running the Web App

Install all the python dependencies:

```
pip install -e .
```

Copy over the local environment configuration file:

```
cp .env.example env  # Update all the variables
```

Start a flask development server:

```
cd aisearch
hypercorn -w 1 -b 0.0.0.0:4000 web.app:app
```
