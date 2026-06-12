# Teacher-Confidence Weighted OPD
next-token entropy:
teacher 觉得“既然你已经走到这里，下一步我是否明确？”
这一个是teacher model对于下一步我是否明确？的计算
## Motivation

Standard OPD asks the teacher to provide token-level supervision on every student-generated prefix:

```text
s_t = (x, y_<t)
```

This assumes the teacher signal is equally useful on all student-visited states. In practice, when the student prefix drifts away from the teacher's familiar reasoning path, the teacher distribution can become high-entropy and less reliable. Teacher-confidence weighted OPD downweights those uncertain prefixes while preserving the original OPD objective as the default behavior.

## Method

Original top-k OPD uses the token-level advantage:

```text
A_t(v) =
[log pi_T(v | s_t) - log pi_S(v | s_t)] * w_t(v)
```

where `w_t(v)` is the configured token weight, such as `student_p`.

The confidence-weighted version adds a prefix-level scalar:

```text
A_t(v) =
c_t * [log pi_T(v | s_t) - log pi_S(v | s_t)] * w_t(v)
```

The current implementation supports:

```text
c_t = exp(-alpha * H_T(s_t))
```

where:

```text
H_T(s_t) = - sum_v pi_T(v | s_t) log pi_T(v | s_t)
```

Lower teacher entropy means higher confidence and stronger OPD signal. Higher teacher entropy means lower confidence and weaker OPD signal.

For stable reward scale, the confidence weights can be normalized by their batch mean and clipped:

```text
c_t <- c_t / mean(c_t)
c_t <- clamp(c_t, c_min, c_max)
```

## Files Changed

- `on_policy_distillation.sh`
  - Adds teacher-confidence parameters with defaults that keep original OPD unchanged.
- `on_policy_distillation_teacher_conf.sh`
  - New wrapper script that enables teacher-confidence weighted OPD.
- `verl/verl/trainer/ppo/ray_trainer.py`
  - Passes teacher-confidence parameters into `batch.meta_info`.
- `verl/verl/workers/actor/dp_actor.py`
  - Multiplies OPD `rm_scores` by teacher confidence when enabled.

## Parameters

| Parameter | Default in original OPD | Default in teacher-confidence script | Meaning |
| --- | --- | --- | --- |
| `TEACHER_CONF_WEIGHT_MODE` | `none` | `entropy_exp` | `none` disables weighting; `entropy_exp` uses `exp(-alpha * entropy)`. |
| `TEACHER_CONF_ALPHA` | `0.2` | `0.2` | Controls how strongly entropy suppresses the reward. |
| `TEACHER_CONF_MIN` | `0.2` | `0.2` | Lower bound after normalization. |
| `TEACHER_CONF_MAX` | `2.0` | `2.0` | Upper bound after normalization. |
| `TEACHER_CONF_NORMALIZE` | `True` | `True` | Keeps the average confidence near 1 on valid response tokens. |

## How To Run

Original OPD:

```bash
bash on_policy_distillation.sh
```

Teacher-confidence weighted OPD:

```bash
bash on_policy_distillation_teacher_conf.sh
```

Override confidence parameters from the command line:

```bash
TEACHER_CONF_ALPHA=0.1 \
TEACHER_CONF_MIN=0.1 \
TEACHER_CONF_MAX=3.0 \
bash on_policy_distillation_teacher_conf.sh
```

Disable the method while using the wrapper:

```bash
TEACHER_CONF_WEIGHT_MODE=none bash on_policy_distillation_teacher_conf.sh
```

## Suggested Ablations

Run these with the same student, teacher, dataset, seed, and training budget:

```text
1. Original OPD:
   TEACHER_CONF_WEIGHT_MODE=none

2. Mild confidence weighting:
   TEACHER_CONF_WEIGHT_MODE=entropy_exp
   TEACHER_CONF_ALPHA=0.1

3. Default confidence weighting:
   TEACHER_CONF_WEIGHT_MODE=entropy_exp
   TEACHER_CONF_ALPHA=0.2

4. Strong confidence weighting:
   TEACHER_CONF_WEIGHT_MODE=entropy_exp
   TEACHER_CONF_ALPHA=0.5
```

Recommended metrics to compare:

```text
- validation accuracy
- overlap ratio
- overlap-token advantage
- student entropy
- teacher entropy
- gradient norm
- training stability at later tokens
```

## Interpretation

If this method helps, it suggests that teacher uncertainty on off-path student prefixes was adding noisy or weakly exploitable token-level gradients. If it hurts, the teacher entropy may be suppressing useful exploration or reducing the dense supervision that makes OPD sample-efficient.
