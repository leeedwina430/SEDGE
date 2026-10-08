# Multimodal discrete candidates, alongside previous benchmarks

Independent roots X1,C,X3 ~ U[-1,1]. Z1=1 iff C is in A=[-1,-.4] union [.4,1] and X1 in its compatible private window. Z2=1 iff C is in B=[-.5,.5] and X3 in its compatible private window. Hold out exactly the (1,1) combination. The individually feasible bridge sets have lengths 1.2 and 1; their intersection is [-.5,-.4] union [.4,.5], two width-.1 intervals. Joint/individual-minimum width ratio .2. No root-space selection is added.

Three variants:

- Balanced: both private windows centered at zero, length w; true overlap modes have equal posterior mass.
- Weighted: first window length w; second length .2w for C<0 and w for C>=0. True overlap mode weights are 1:5, not 1:1.
- Curved: private centers .45 sin(pi C) and .45 sin(2 pi C), lengths w(.6+.4C²) and w(.2+.8((C+1)/2)²). Conditional posterior density varies within modes; right-mode probability approximately .704399.

All private windows lie within [-1,1]. Two declared regimes use w=.08 (strict conditional exclusion <=1%) or w=.5 (reference-like relaxed controls, <=5%, NOT claimed 1%-compliant). Shrinking private windows changes finite positive-label counts as well as selection bias, so these regimes are not an isolated intervention on Assumption 3.

| Variant | Strict P(S=0|Z1=1) / P(S=0|Z2=1) | Reference probabilities | Right-mode posterior probability |
|---|---|---|---:|
| Balanced | .667% / .800% | 4.167% / 5.000% | .5000 |
| Weighted | .400% / .800% | 2.500% / 5.000% | .8333 |
| Curved | .248% / .577% | 1.549% / 3.604% | .7044 |

selection_audit.json records exact private-window integration, numerical quadrature, support sets, excluded mass (diagnostic only), mode probabilities, and example training counts. All six passed their explicitly declared thresholds before submission. Exact reference generation draws C in the overlap with density proportional to the product of private-window lengths, then draws each private root uniformly in its own window. No oracle coefficients or labels train the learned likelihood; the known conditional Bernoulli probability is used only in the labeled oracle-weight and fit diagnostics.

Same full suite as ../sedge_benchmark: CD; compositional CD; three legacy adaptations (mean/residual); SEDGE-LWS; SEDGE-LGD; dense-anchor LGD; sparse/dense factorized and joint CG; uniform and oracle controls. Same 20,000 fitting/2,000 validation rows, diffusion/density budgets, 2,000 test draws, guidance scales and proposal budgets. Binary likelihoods are Bernoulli, not a continuous flow on discrete labels; no GMM training.

Numerical exception: smoke testing exposed undefined least-squares gradients in the Rep4Ex-inspired adaptation when a rare-positive minibatch is rank-deficient. This extension adds ridge 1e-4 to its 3x3 affine Gram matrix before solving. The change is disclosed rather than silently called an identical implementation; historical results remain unchanged. It stabilizes the regression, not the sampled outputs, and no test information enters the solve.

Six cases x three seeds = 18 complete suites. Mode-mass error and separate valid left/right masses augment all existing validity/MMD/NN/likelihood metrics. Mode fractions conditional on overlap/validity are explicitly labeled; invalid outputs stay in the overall denominator for validity. Old results are never replaced. COMPARISON_ALL.md and all_results.json combine these cases with all completed prior progressive-suite cases. All other historical folders remain available unchanged.

Exploratory candidates: no guarantee that SEDGE stays >=85% or that compositional CD fails. Report every case and seed. Later benchmark selection should be confirmed with fresh independent seeds rather than portrayed as a preregistered comparison.
