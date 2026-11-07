# Contributing

Thanks for looking into OTSR. The project is small and the rules are few:

1. **The Python reference is the specification.** A change to the maths goes into `reference/otsr_ref/` first,
   then into the TypeScript / WGSL, then `npm run golden` regenerates `tests/golden/` and `npm run test:gpu` must
   pass. Never edit golden files by hand.
2. **Every kernel has a test** against the reference or against an exact invariant (see `docs/testing.md`).
3. **Keep it practical.** The engine favours one clear implementation over options; please discuss larger features
   in an issue before building them.
4. `npm run typecheck`, `npm test`, `npm run test:gpu` and `npm run smoke` (after `npm run build`) are the checks a
   pull request should pass. The GPU suites need a machine with a GPU; say so in the PR if you could not run them.

Setup, layout and conventions are described in `docs/development.md`.
