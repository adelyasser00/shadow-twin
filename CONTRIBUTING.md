# Contributing

## Before you write code

Run both self-test suites. They check the solver against values you can derive
on paper, and they are the reason anything here can be trusted.

```
python -m pipeline.verify_solar
python -m pipeline.verify_geometry
```

Both must print `ALL CHECKS PASSED`. If either fails on a clean checkout, that
is a bug worth reporting on its own.

## The rules that are not negotiable

**Never make a failing test pass by changing the test.** A shadow length of
`H / tan(altitude)` is arithmetic, not a matter of opinion. If `verify_geometry`
fails, the solver is wrong.

**Never assume units.** NYC LiDAR is US survey feet on every axis. The readers
detect the unit from the CRS and refuse to guess. Reading feet as metres makes
every shadow 3.28 times too long, and the output still looks plausible, which is
what makes it dangerous.

**`geometry.py` is the reference.** `geometry_torch.py` must match it cell for
cell, and `verify_torch.py` checks that. Where they disagree, the CPU path wins.

**Fixed colour bins.** Layers are binned on stated thresholds and never
stretched to their own range. Two runs have to stay comparable.

**Do not weaken the caveats.** The About panel lists what the model does not
compute. Adding a capability means updating that list. Removing a caveat to make
the output sound stronger is not a contribution.

## Scope

This computes shadow geometry. It does not compute temperature, wind, or thermal
comfort, and a pull request that implies otherwise in the UI copy will be asked
to change the copy, not the claim.
