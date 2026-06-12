# Teacher-Prefix-Likelihood Weighted OPD

## Motivation

Standard OPD distills the teacher on every student-generated prefix:

```text
s_t = (x, y_<t)
```

However, some student prefixes may be paths that the teacher itself would almost never take. In those states, forcing the student to match the teacher's next-token distribution can be less useful, because the teacher is being queried on an off-path reasoning state.

Teacher-Prefix-Likelihood Weighted OPD asks a different question:

```text
Does the teacher recognize the current student prefix as a plausible teacher-like path?
```

If yes, distill strongly. If no, downweight the OPD signal.

## Prefix Likelihood

For a student-generated response:

```text
y = (y_1, ..., y_T)
```

the teacher likelihood of the prefix before position `t` is:

```text
log P_T(y_<t | x)
= sum_{i=1}^{t-1} log P_T(y_i | x, y_<i)
```

To avoid length bias, we use average token log-likelihood:

```text
ell_t =
1 / (t - 1) * sum_{i=1}^{t-1} log P_T(y_i | x, y_<i)
```

Then the prefix confidence is:

```text
c_t = exp(alpha * ell_t)
```

When `alpha = 1`, `c_t` is the geometric mean teacher probability of the prefix tokens. Because `ell_t <= 0`, larger values mean the teacher assigns higher probability to the prefix.

For `t = 1`, there is no response prefix yet, so the implementation sets:

```text
c_1 = 1
```

## Weighted OPD Objective

Original OPD:

```text
L_OPD =
sum_t KL(P_S(. | x, y_<t) || P_T(. | x, y_<t))
```

Prefix-likelihood weighted OPD:

```text
L =
sum_t c_t KL(P_S(. | x, y_<t) || P_T(. | x, y_<t))
```

In the top-k implementation, this is applied to the token-level advantage:

```text
A_t(v) =
c_t * [log P_T(v | x, y_<t) - log P_S(v | x, y_<t)] * w_t(v)
```

where `w_t(v)` is the existing OPD token weight, such as `student_p`.

## Difference From Entropy Confidence

This method is different from `entropy_exp`.

Entropy confidence asks:

```text
Given this prefix, is the teacher certain about the next token?
```

Prefix-likelihood confidence asks:

```text
Would the teacher have reached this prefix in the first place?
```

So the two can behave differently. A strange prefix may still lead to a low-entropy next-token distribution, but prefix likelihood can still downweight it because the path itself was unlikely under the teacher.

## Implementation

The implementation uses the teacher's log-probability of the actual student-sampled tokens:

```text
teacher_sample_log_probs[t]
= log P_T(y_t | x, y_<t)
```

For each position, it computes the cumulative mean over previous tokens:

```text
ell_t =
mean(teacher_sample_log_probs[1 : t - 1])
```

Then:

```text
c_t = exp(alpha * ell_t)
```

The confidence weights are optionally normalized and clipped:

```text
c_t <- c_t / mean(c_t)
c_t <- clamp(c_t, c_min, c_max)
```

This keeps the overall reward scale close to original OPD while still redistributing emphasis across positions.

## Files

- `on_policy_distillation_prefix_likelihood.sh`
  - Enables this method.
- `on_policy_distillation.sh`
  - Carries the shared OPD configuration and confidence parameters.
- `verl/verl/workers/fsdp_workers.py`
  - Returns `teacher_sample_log_probs`.
- `verl/verl/workers/actor/dp_actor.py`
  - Computes prefix-likelihood confidence and applies it to OPD `rm_scores`.
- `verl/verl/trainer/ppo/ray_trainer.py`
  - Passes config into `meta_info` and logs confidence statistics.

## Parameters

| Parameter | Default | Meaning |
| --- | --- | --- |
| `TEACHER_CONF_WEIGHT_MODE` | `prefix_likelihood` | Enables prefix-likelihood weighting. |
| `TEACHER_CONF_ALPHA` | `1.0` | Exponent scale for average teacher prefix log-likelihood. |
| `TEACHER_CONF_MIN` | `0.2` | Minimum confidence after normalization. |
| `TEACHER_CONF_MAX` | `2.0` | Maximum confidence after normalization. |
| `TEACHER_CONF_NORMALIZE` | `True` | Normalize confidence by valid-token mean. |

## How To Run

Original OPD:

```bash
bash on_policy_distillation.sh
```

Teacher-prefix-likelihood weighted OPD:

```bash
bash on_policy_distillation_prefix_likelihood.sh
```

Override parameters:

```bash
TEACHER_CONF_ALPHA=0.5 \
TEACHER_CONF_MIN=0.1 \
TEACHER_CONF_MAX=3.0 \
bash on_policy_distillation_prefix_likelihood.sh
```

## Suggested Ablations

Use the same student, teacher, dataset, response length, and seed:

```text
1. Original OPD:
   TEACHER_CONF_WEIGHT_MODE=none

2. Prefix-likelihood OPD:
   TEACHER_CONF_WEIGHT_MODE=prefix_likelihood
   TEACHER_CONF_ALPHA=1.0

3. Milder prefix weighting:
   TEACHER_CONF_WEIGHT_MODE=prefix_likelihood
   TEACHER_CONF_ALPHA=0.5

4. Stronger prefix weighting:
   TEACHER_CONF_WEIGHT_MODE=prefix_likelihood
   TEACHER_CONF_ALPHA=2.0
```

Track:

```text
- validation accuracy
- overlap ratio
- overlap-token advantage
- teacher_confidence/mean
- teacher_confidence/std
- teacher entropy
- student entropy
- gradient norm
```

## Optional Future Extension: Path Penalty

This implementation only reweights OPD. It does not directly penalize low-likelihood student paths.

A stronger extension is:

```text
A_path,t = c_{t+1} - b
L_path = - sum_t stopgrad(A_path,t) log P_S(y_t | x, y_<t)
```

This would:

```text
c_{t+1} > b  -> increase probability of y_t
c_{t+1} < b  -> decrease probability of y_t
```

That path penalty is intentionally not included in the current implementation, because the current OPD top-k loss optimizes a 3D token set `(batch, seq, k)`, while path penalty is a sampled-token 2D policy-gradient term. Keeping it separate makes the first ablation cleaner.
