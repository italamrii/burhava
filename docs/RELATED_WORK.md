# Related work

One line each on how the work overlaps with Burhava. Entries marked **⚠ unverified** came from the project brief. I have not yet checked their arXiv IDs, titles or content, so they must be verified before any citation is used in the paper.

## Value of information and sequential testing (the method's foundations)
* **Howard 1966**, "Information Value Theory" (IEEE Trans. Systems Science and Cybernetics). The origin of VoI. Our `Δ_i(S)` is exactly this quantity applied to verification.
* **Krause & Guestrin 2009**, "Optimal Value of Information in Graphical Models" (JAIR). Exact and approximate VoI-based observation selection in graphical models, including hardness results. This is our selection rule in its general form.
* **Golovin & Krause 2011**, "Adaptive Submodularity" (JAIR). Guarantees for adaptive greedy when the objective is adaptive submodular. Decision-risk VoI is generally *not* submodular, as the AND counterexample in MATH.md §5.1 shows, so these guarantees do not transfer directly.
* **Chen et al. 2015**, "Submodular Surrogates for Value of Information" (AAAI; the DiRECt method). Replaces non-submodular VoI with a submodular surrogate for decision-making. This is a direct alternative to our lookahead fix for greedy stalls.
* **Deshpande, Hellerstein & Kletenik 2014**, "Approximation Algorithms for Stochastic Boolean Function Evaluation and Stochastic Submodular Set Cover" (SODA). Evaluates a Boolean function of costly random bits at minimum expected cost. Our problem has the same structure (the conclusion is a Boolean function of atoms), but it stops on decision risk rather than requiring certainty.
* **Heckerman, Breese & Rommelse 1995**, "Decision-Theoretic Troubleshooting" (CACM). Sequences costly observations and repairs by expected cost. This is classical precedent for cost-aware test selection.
* **Van den Broeck et al. 2010**, "DTProbLog: A Decision-Theoretic Probabilistic Prolog" (AAAI). Decision-making over probabilistic logic programs. Our AND/OR graphs are a small special case of such programs, and ProbLog is an optional later backend.

## LLM verification and claim-level confidence (⚠ unverified, from brief)
* **Best-of-Evidence**, arXiv 2607.20950. ⚠ unverified: overlap to be summarised after reading.
* **Entropic Claim Resolution**, arXiv 2603.28444. ⚠ unverified. From the title, it likely selects claims by entropy, which would make it the closest relative of arm 4 (highest-entropy-first).
* **Graph of Verification**, arXiv 2506.12509. ⚠ unverified. From the title, it likely verifies reasoning as a graph, which overlaps with our dependency-graph recovery in Stage C.
* **Claim-Level Confidence Calibration**, arXiv 2608.22483. ⚠ unverified. Likely relevant to Stage B/C calibration of per-premise confidences.

## Benchmarks (Stage C)
* **RuleTaker**: Clark, Tafjord & Richardson 2020, "Transformers as Soft Reasoners over Language" (IJCAI). Synthetic rule-based reasoning with gold proofs.
* **ProofWriter**: Tafjord, Dalvi & Clark 2021 (Findings of ACL). Extends RuleTaker with proofs and closed/open-world negation. This is one reason our graphs support negated premises.
* **PrOntoQA**: Saparov & He 2023, "Language Models Are Greedy Reasoners" (ICLR). Synthetic ontology-based chains with gold proofs.
