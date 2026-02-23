# GNAIS

## Description

**GNAIS** (GeneNetwork AI Search) is a python package that helps digest metadata around GeneNetwork using language models. It allows running natural language queries against RDF data (metadata) converted to text and preprocessed locally.

**GNAIS** performs a hybrid search (keyword and semantic) through a RAG (Retrieval Augmented Generation) system. The embedding model for semantic is Qwen/Qwen3-Embedding-0.6B (open model).

We implemented **GNAIS** using [DSPy](https://dspy.ai/). Switching between LLM providers for the text generation model is as easy as changing a variable :)

## Installation

**GNAIS** is in PyPI. You can install it in your virtual environment using the following commands:

```python
python -m venv .venv
source .venv/bin/activate
pip install gnais
```

## Usage

To use **GNAIS**, you need to define a few parameters from your bash environment.

```bash
export CORPUS_PATH=<YOUR_PATH>
export PCORPUS_PATH=<YOUR_PATH>
export DB_PATH=<YOUR_PATH>
export SEED=<YOUR_VALUE>
export MODEL_TYPE=<YOUR_VALUE>
export MODEL_NAME=<DSPY_COMPLIANT_MODEL_NAME>
export API_KEY=<YOUR_API_KEY_IF_REQUIRED>
export QUERY=<YOUR_QUERY>
```

Once defined, you can run your search from a Python shell or script with:

```python
from gnais.search import search, QUERY
from gnais.search import *

search(QUERY)
```

## Examples

Here are a few examples of queries and responses generated with gnais using claude-haiku-4-5-20251001 as core model.

*Query 1: Digest and explain phenotype datasets around cognition measured in BXD mice*

Response:
```
{
    "detailed_answers": [
        {
            "answer": "Age at onset of working memory deficits (AAO_NtgBXD) is a cognition-related phenotype measured in non-transgenic BXD littermates from the AD-BXD population. This trait measures the age at which individual mice show impaired working memory performance (defined as chance performance dropping below 50%). The mean value is 11.12 months with a LOD score of 1.89.",                                                                                                                                                                          
            "links": [
                "https://cd.genenetwork.org/show_trait?trait_id=20556&dataset=BXDPublish",
                "http://rdf.genenetwork.org/v1/id/set_BXDPublish"
            ]
        },
        {
            "answer": "Age at onset of working memory deficits in males (AAO_NtgBXD_Males) measures working memory impairment onset specifically in male non-transgenic BXD littermates from the AD-BXD population. The mean age is 10.56 months with a LOD score of 4.57, indicating a stronger genetic signal than the combined sex trait.",                                   
            "links": [
                "https://cd.genenetwork.org/show_trait?trait_id=20884&dataset=BXDPublish",
                "http://rdf.genenetwork.org/v1/id/set_BXD"
            ]
        },
        {
            "answer": "Age at onset of working memory deficits in females (AAO_NtgBXD_Females) measures working memory impairment onset specifically in female non-transgenic BXD littermates from the AD-BXD population. The mean age is 11.70 months with a LOD score of 2.06.",                                                                                               
            "links": [
                "https://cd.genenetwork.org/show_trait?trait_id=20803&dataset=BXDPublish",
                "http://rdf.genenetwork.org/v1/id/set_BXD"
            ]
        },
        {
            "answer": "Impulsivity measured as motor impulsivity in the 5-choice serial reaction time task (5CSRT) is a cognition-related phenotype in male BXD mice. It measures premature responses before stimulus presentation, with higher values indicating greater impulsivity and poorer performance. The mean value is 9.35 with a LOD score of 3.60.",                 
            "links": [
                "https://cd.genenetwork.org/show_trait?trait_id=16311&dataset=BXDPublish",
                "http://rdf.genenetwork.org/v1/id/set_BXD",
                "http://rdf.ncbi.nlm.nih.gov/pubmed/24703509"
            ]
        }
    ],
    "final_answer": "The BXD mouse phenotype datasets contain several cognition-related measurements. The primary focus is on working memory deficits, with three related traits measuring age at onset (AAO) of working memory impairment in non-transgenic BXD littermates from the AD-BXD population: a combined sex-averaged trait (mean 11.12 months, LOD 1.89), a male-specific trait (mean 10.56 months, LOD 4.57), and a female-specific trait (mean 11.70 months, LOD 2.06). These traits are part of a larger study examining cognitive aging in transgenic and non-transgenic AD-BXD mice. Additionally, the dataset includes a motor impulsivity trait measured via the 5-choice serial reaction time task (5CSRT) in males, which assesses premature responding behavior (mean 9.35, LOD 3.60). All cognition traits were submitted by Sarah Neuner and Catherine Kaczorowski, and the working memory traits are referenced in unpublished studies, while the impulsivity trait is published in PubMed:24703509."                                                                                                                       
}
```


*Query 2: Compare traits related to the central nervous system measured in BXD, BDL, CCB and MDP mice*


```
{
    "detailed_answers": [
        {
            "answer": "BXD traits related to central nervous system include: BXD_21300 (CMS_Sacrifie_Age - age at sacrifice after CMS treatment, mean: 166.93), BXD_21301 (NH_Sacrifie_Age - age at sacrifice in normally housed mice, mean: 172.60), BXD_21294 (CMS_Prefrontal_Cortex_W_1000_BW - prefrontal cortex weight/BW ratio, mean: 0.876), BXD_21292 (NH_HW_BW - heart weight/BW ratio, mean: 0.499), BXD_21291 (CMS_HW_BW - heart weight/BW ratio after CMS treatment, mean: 0.517), BXD_21296 (CMS_Hypotalamus_W_1000_BW - hypothalamus weight/BW ratio, mean: 0.369), BXD_11346 (M_NEPDIST30 - locomotion in periphery 15-30 min, mean: 2891.44 cm), BXD_11347 (M_NEPDIST45 - locomotion in periphery 30-45 min, mean: 2386.69 cm), BXD_11342 (M_NEPCOUNT30 - locomotion beam breaks 15-30 min, mean: 796.69), and BXD_21500 (myelinating oligodendrocytes proportion difference, mean: 29.32).",
            "links": [
                "https://cd.genenetwork.org/show_trait?trait_id=21300&dataset=BXDPublish",
                "https://cd.genenetwork.org/show_trait?trait_id=21301&dataset=BXDPublish",
                "https://cd.genenetwork.org/show_trait?trait_id=21294&dataset=BXDPublish",
                "https://cd.genenetwork.org/show_trait?trait_id=21292&dataset=BXDPublish",
                "https://cd.genenetwork.org/show_trait?trait_id=21291&dataset=BXDPublish",
                "https://cd.genenetwork.org/show_trait?trait_id=21296&dataset=BXDPublish",
                "https://cd.genenetwork.org/show_trait?trait_id=11346&dataset=BXDPublish",
                "https://cd.genenetwork.org/show_trait?trait_id=11347&dataset=BXDPublish",
                "https://cd.genenetwork.org/show_trait?trait_id=11342&dataset=BXDPublish",
                "https://cd.genenetwork.org/show_trait?trait_id=21500&dataset=BXDPublish"
            ]
        },
        {
            "answer": "BDL (BXD Longevity) traits related to central nervous system include: BDL_10091 (Baseline_response - contextual fear conditioning baseline freezing in aging BXD males and females, mean: 4.54%). This trait measures baseline anxiety in aging mice on different diet conditions (chow vs high fat).",
            "links": [
                "https://cd.genenetwork.org/show_trait?trait_id=10091&dataset=BXD-LongevityPublish"
            ]
        },
        {
            "answer": "CCB (CCBXD_TM) traits related to central nervous system include: CCB_10022 (fear_acq - fear acquisition/learning, mean: 58.10%), CCB_10023 (fear_ext - fear extinction/memory, mean: 21.77%), CCB_10024 (fear_exp - fear expression/memory, mean: 75.47%), CCB_10057 (bof_prct_cntr_dist - anxiety-like behavior in bright open field, mean: 12.82%), CCB_10079 (dsi_aggr_dur - aggression duration in social interaction, mean: 5.00 s), CCB_10082 (dsi_sniff_dur - sniffing duration in social interaction, mean: 64.05 s), CCB_10077 (SD_between_score - social dominance over strangers, mean: 0.502), and CCB_10058 (SD_within_score - social dominance over cagemates, mean: 0.501).",
            "links": [
                "https://cd.genenetwork.org/show_trait?trait_id=10022&dataset=CCBXD-TMPublish",
                "https://cd.genenetwork.org/show_trait?trait_id=10023&dataset=CCBXD-TMPublish",
                "https://cd.genenetwork.org/show_trait?trait_id=10024&dataset=CCBXD-TMPublish",
                "https://cd.genenetwork.org/show_trait?trait_id=10057&dataset=CCBXD-TMPublish",
                "https://cd.genenetwork.org/show_trait?trait_id=10079&dataset=CCBXD-TMPublish",
                "https://cd.genenetwork.org/show_trait?trait_id=10082&dataset=CCBXD-TMPublish",
                "https://cd.genenetwork.org/show_trait?trait_id=10077&dataset=CCBXD-TMPublish",
                "https://cd.genenetwork.org/show_trait?trait_id=10058&dataset=CCBXD-TMPublish"
            ]
        },
        {
            "answer": "MDP (Mouse Diversity Panel) traits related to central nervous system include: MDP_49906 (LOCACTChrnEtOH_SENZ - ethanol sensitization measured as difference in locomotion after chronic ethanol treatment, mean: 659.67 beam breaks/15 min). This trait measures the sensitization response to repeated ethanol exposure.",
            "links": [
                "https://cd.genenetwork.org/show_trait?trait_id=49906&dataset=MDPPublish"
            ]
        },
        {
            "answer": "Comparison summary: BXD traits focus on pharmacology (CMS stress response), morphology (brain region weights), and locomotor behavior. BDL traits measure aging-related behavioral responses (fear conditioning). CCB traits emphasize behavioral phenotypes including cognition (fear learning/extinction), anxiety, and social behavior. MDP traits measure drug response (ethanol sensitization). All datasets measure central nervous system-related phenotypes but with different emphases: BXD on stress/morphology, BDL on aging, CCB on behavior/cognition, and MDP on pharmacological responses.",
            "links": []
        }
    ],
    "final_answer": "Central nervous system traits across the four mouse datasets show distinct research focuses: BXD mice (10 traits) include stress-induced changes (CMS treatment effects on organ weights and sacrifice age), morphological measurements (brain region weights), and locomotor behavior in novel environments. BDL mice (1 trait) measure aging-related fear conditioning and baseline anxiety responses. CCB mice (8 traits) emphasize behavioral and cognitive phenotypes including fear learning/extinction, anxiety-like behavior, and social interaction measures (dominance, aggression, sniffing). MDP mice (1 trait) focus on pharmacological responses, specifically ethanol sensitization. The BXD dataset provides the most comprehensive CNS phenotyping with 10 traits covering pharmacology, morphology, and behavior. CCB traits are particularly rich in behavioral/cognitive measures, while BDL and MDP datasets each contribute specialized phenotypes for aging and drug response respectively. All traits are measured in adult or aging mice, with BXD and CCB using controlled behavioral testing paradigms and BXD additionally measuring stress-induced physiological changes."
}
```
