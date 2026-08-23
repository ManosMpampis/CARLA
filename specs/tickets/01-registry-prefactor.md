# 01: Registry prefactor — legacy backbone rename + factory registry

**What to build:** Make the change easy first: the typo-named legacy backbone module is renamed with all imports updated, and model/criterion construction moves behind registry factories keyed by config names — the repo's existing factory pattern, now the single wiring point every later arm plugs into. Pure prefactor; zero behavior change.

**Blocked by:** None (can start immediately).

**Status:** ready-for-agent

- [ ] Legacy backbone importable under its corrected name; old typo name gone
- [ ] Existing pretext config still instantiates its model end-to-end through the registry (legacy entrypoint unaffected)
- [ ] No functional change: same weights, same outputs for a fixed seed
