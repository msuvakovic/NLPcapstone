# Gemma output-token cap experiment

Source-only tuning: 24 balanced SST-2 training examples for selection and 64 separate SST-2 validation examples for confirmation. No final source or target tests were used to select a cap. The base instruction and decoding settings are fixed.

## Selection (24 examples per cap)

| Cap | Accuracy | Macro F1 | Invalid | Length finishes | Agreement with 16 | Mean output tokens | Median latency | p95 latency |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 91.67% | 0.9161 | 0 | 24 | 24/24 | 1.00 | 2.393s | 16.590s |
| 2 | 91.67% | 0.9161 | 0 | 24 | 24/24 | 2.00 | 2.427s | 11.880s |
| 4 | 91.67% | 0.9161 | 0 | 0 | 24/24 | 2.00 | 2.408s | 6.259s |
| 8 | 91.67% | 0.9161 | 0 | 0 | 24/24 | 2.00 | 2.401s | 12.133s |
| 16 | 91.67% | 0.9161 | 0 | 0 | 24/24 | 2.00 | 2.499s | 6.024s |

## Confirmation (64 examples per cap)

| Cap | Accuracy | Macro F1 | Invalid | Length finishes | Agreement with 16 | Mean output tokens | Median latency | p95 latency |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 96.88% | 0.9687 | 0 | 64 | 64/64 | 1.00 | 2.518s | 12.005s |
| 4 | 96.88% | 0.9687 | 0 | 0 | 64/64 | 2.00 | 2.461s | 11.304s |
| 16 | 96.88% | 0.9687 | 0 | 0 | 64/64 | 2.00 | 2.457s | 13.427s |

## Interpretation

**For the exact tested classifier, cap 1 is the observed minimum-token setting; cap 4 is an alternative when a natural stop is required.** Cap 1 reduced output tokens by 50% and total input-plus-output tokens by 1.54% on the confirmation sample. Median latency was 2.52s versus 2.46s at cap 16, so no latency improvement was demonstrated.

Preselected smallest complete-label cap: **1**. Confirmation matched all reference labels with no invalid outputs: **True**.
Preselected smallest tested cap allowing natural stopping on the selection sample: **4**. See its confirmation length-finish count above. Cap 3 was not tested.

A `length` finish means the token ceiling was reached; a complete Positive/Negative label can still be valid. Partial labels are invalid and count as errors. The table reports these separately.

A cap is an upper bound, not the number of tokens always generated. Reducing 16 to 4 or 8 saves nothing when an answer already stops at two tokens. Input tokens are unchanged. Token reductions are not verified billing savings.

These results cover one fixed sentiment prompt and one model endpoint. The small source-only confirmation cannot establish rare-failure rates or OOD accuracy. Provider caching and load were uncontrolled; single-request latency excludes failed attempts and is descriptive. Cached resumes do not create independent measurements.

Do not apply a one-word classification cap to prompt proposals, reflections, NLI labels, explanations or reasoning tasks. Existing frozen benchmark settings and results remain unchanged.
