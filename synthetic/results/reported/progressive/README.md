# Audited progressive SEDGE benchmark

## Selection before training

The mandatory original discrete square benchmark is retained as an explicitly marked historical reference: P(S=0|Zi=1)=5%, NOT compliant with the working 1% threshold. A stricter discrete square variant has 0.9% conditional exclusion. Every continuous candidate satisfies the numerical <=1% condition for BOTH specification bands at two resolutions. Overall excluded probability is reported, not constrained. See selection_audit.json for full geometry, probability and resolution records. No model performance enters the selection. None of these geometry checks guarantees >=85% validity or a particular separation from CD.

| Benchmark | Individual bridge ranges at target center | Joint bridge range | Conditional exclusion Z1 / Z2 |
|---|---|---|---|
| Discrete squares, reference | [0,1]; [-.9,.1] | [0,.1] | 5% / 5% (exception) |
| Discrete squares, strict | [0,1]; [-.982,.018] | [0,.018] | .9% / .9% |
| Linear | [0,.525]; [.475,1] | [.475,.525] | approximately .0357% / .0357% |
| Cubic | [0,.525]; [.475,1] | [.475,.525] | approximately .0357% / .0357% |
| Smoothstep | [0,.525]; [.475,1] | [.475,.525] | approximately .0357% / .0357% |
| Private-bridge interaction | [0,.525]; [.475,1] | [.475,.525] | approximately .0291% / .0291% |
| Two-dimensional bridge | coordinate projections [0,.5]x[0,1]; [0,1]x[.5,1] | projections [0,.125]x[.875,1] | approximately .0116% / .0116% |

The 2D feasible sets are NOT the Cartesian products of their projections. Individual feasible area is .2836–.2879 and joint area .00840–.00947 across query pairs; the full inequalities and all query projection ranges are in data.py/selection_audit.json. Scalar joint width varies around .05 across the square; exact extrema are audited.

## DGPs

Binary roots are independent U[-1,1]. Both private intervals remain [0,1] and bridge intervals have unit width, so both specification regions are genuinely squares. Only bridge overlap changes.

Continuous roots are independent U[0,1]. Linear/cubic/smoothstep: Z1=.8X1+.6C, Z2=.8X3+.6k(C), with k(C)=C, C^3, or 3C^2-2C^3. Interaction: Z1=C+.8(1+.3C)X1; Z2=1-C+.8(1+.3(1-C))X3. Two-dimensional: Z1=.7C1+.3C2+(.8+.15C1C2)X1; Z2=1-.3C1-.7C2+(.8+.15(1-C1)(1-C2))X3. The last two introduce private/bridge interactions; the last adds nonproportional shared parents.

The selected rectangle itself is removed; no extra guard. Every remaining marginal has a support witness outside the other band's exclusion. Exact posterior reference sampling uses rejection in (Z,C) weighted by the inverse-private Jacobian 1/(a1*a2), not uniform Z or uniform C when scales vary. All private scales >=.8, validating the rejection bound. Repeated-reference equations and root support are asserted.

## Models and formulations

- CD: full conditional diffusion p(X|Z1,Z2).
- Compositional CD: score s(Xt|Z1)+s(Xt|Z2)-s(Xt), with individually trained conditions and a shared prior. This noisy-state composition is approximate; clean CI does not imply noisy CI.
- Additive, fixed-soft-composition, Rep4Ex-inspired adaptations: reuse the historical implementation, both mean and residual-sampling outputs. Rep4Ex remains an adaptation with documented proxy/identifiability limitations, not a theorem replication. For 2D bridge, input/output X dimension is increased from 3 to 4; latent representation stays 2-dimensional and is disclosed.
- SEDGE-LWS: anchor p(X1,C|Z1), weights p(Z2|C), private completion p(X3|C,Z2). Uniform and oracle-weight controls share the proposal bank, uniforms and private noise. Oracle controls never train the model.
- SEDGE-LGD: same factorization. Distill fresh samples from the learned clean flow/Bernoulli likelihood into q(Z2|Ut,Z1,t), where U=(C,X1), then guide the anchor diffusion. This uses the approximate learned-factor joint, not a clean-likelihood gradient mistakenly called noisy CG.
- SEDGE-dense-LGD (no sparsity): replace the anchor by full p(X|Z1), guide q(Z2|Xt,Z1,t), no private completion. Its noisy density is trained on observed pairs because noiseless p(Z2|X) is singular; no ordinary clean flow is claimed. This differs in density supervision from teacher-distilled sparse LGD, so it is not a perfectly isolated parameter-mask intervention.
- Matched CG controls: same prior for sparse-factorized noisy likelihood, dense-factorized likelihood, and dense-joint autoregressive likelihood. Sparse->dense tests parent restrictions at fixed sampler. Factorized->joint tests noisy-state CI, not the truth of clean-state CI. Joint Bernoulli is autoregressive for discrete Z; continuous joint density is an autoregressive pair of conditional spline flows.

All continuous likelihoods are flows; no GMM is trained. Clean SEDGE flow uses training-fitted polynomial lower/upper support, degree up to 3 plus cross term for 2D. This is an explicit inductive bias, not oracle coefficients or a universally suitable support model. Noisy-state flows have Gaussian base and identity tails, without oracle support. Binary likelihoods are Bernoulli.

## Budget and reporting

Seven cases x three seeds = 21 complete suites. 20,000 fitting + 2,000 validation rows. Continuous evaluations: 64 population-distributed test targets x 32 independent draws = 2,048 outputs; binary: 2,000 outputs at the unseen (1,1) combination. Each method sees the same target panel and independent oracle reference. Diffusions: 20,000 updates, validation-selected EMA, width192, historical 100-step DDPM schedule. Density models: 10,000 updates. Legacy adaptations: 3,000 updates. Different method parameter totals are recorded; do NOT claim exact capacity/compute matching. Guidance scales .5,1,2 are all reported, not selected on the test set; gradient norm capped at 50 and disclosed. LWS uses 1,024 proposals, so equal output counts do NOT imply equal compute.

Report support-aware joint validity, both separate Zi satisfaction rates, residual-only satisfaction, bridge overlap, overlap AND validity, X-MMD, joint (X,Z)-MMD, repeated-target conditional MMD, NN coverage and both directed NN distances, diversity, mechanism MAE, clean likelihood mean MAE, density MAE/integrated conditional TV where available, full validation NLL/support failures, ESS/empty and missing-compatible banks, and runtime. NN coverage uses the oracle-reference 95th-percentile leave-one-out distance as its shared radius, not a tuned method-specific threshold. Small repeated panels have finite-sample conditional MMD floors. Detailed likelihood and sampling metrics reside next to the main results, not silently omitted.

Full validation NLL is infinite if a bounded flow excludes any validation point; JSON represents it as null plus a nonzero outside-support count. Checkpoint selection then uses supported validation NLL, explicitly recorded. No unsupported sample is removed from generation validity; undefined full-panel MMD is NA rather than a successful-subset score.

All settings, including those where SEDGE does not outperform CD, will be reported. These new DGPs are exploratory; independent confirmation is needed before claiming a general superiority result. Existing jobs and historical outputs are untouched.
