# BlindDrive

BlindDrive is a small, auditable **partial-observation online driving benchmark**.

A car with inertia must finish a fixed-length random road as fast as possible
while only seeing a limited stretch of road ahead. Every decision is made
online, at a fixed rate, from a local observation. Hit the edge and you lose a lot
of speed (or, in strict `--edge crash` mode, the run is over).

> **The environment owns physics. The controller owns decisions.**

## Core rules

A controller:

- only sees the local road: `lookahead` metres ahead (plus a few metres behind), in car-local coordinates;
- never has access to the future road (the full road lives only inside the environment);
- can only return one of the 35 standard `Action`s;
- decides at `decision_hz` (default 4 Hz = every 250 ms); the last action is held in between.

The environment:

- simulates, observes, and judges finish / crash / timeout — nothing else;
- **never** corrects, overrides or filters an action: no auto-braking, no auto-steering,
  no "keep in lane", no collision avoidance. If the controller is wrong, the car hits the edge.

The human player goes through the exact same interface (`HumanController`), at the
same decision rate, and the screen draws the road only from the `Observation`
(except in `--debug`). A human has no extra information and no extra control rate.

Jev (TypeSafe AI) plugs in through the very same interface: see [Driving with Jev](#driving-with-jev).

## Quick start

```bash
cd blinddrive
uv run python -m blinddrive.runner                    # random seed, normal preset
uv run python -m blinddrive.runner --seed 42
uv run python -m blinddrive.runner --seed 42 --preset hard
uv run python -m blinddrive.runner --seed 42 --debug  # full road visible: development only
uv run python -m blinddrive.runner --fps 144          # render frame-rate cap (default 120, 0 = uncapped)
uv run python -m blinddrive.runner --edge crash       # strict mode: touching the edge ends the run
uv run pytest
```

Presets: `easy`, `normal`, `hard` (see `blinddrive/config.py`).

| preset | lookahead | road width | max speed | tightest curve radius |
|--------|-----------|------------|-----------|-----------------------|
| easy   | 40 m      | 10 m       | 22 m/s    | 30 m                  |
| normal | 30 m      | 8 m        | 30 m/s    | 18 m                  |
| hard   | 20 m      | 6.5 m      | 40 m/s    | 12 m                  |

### Controls

| key | action |
|-----|--------|
| W / Up | accelerate (hold Shift: full accelerate) |
| S / Down | brake |
| Space | hard brake |
| A / Left, D / Right | steer (hold Shift: hard) |
| Q / E | slight left / slight right |
| nothing | coast, straight |
| Enter | start (and retry after a run) |
| R · N · Esc | retry same seed · new seed · quit |

Keys are read **only at decision ticks** (every 250 ms), exactly like any other controller.

### Display

The view is deliberately minimal: the visible road (edges, a dashed centerline,
the finish line when in sight), the car (a bar at its front tilts with the
wheels; the car turns red while scraping an edge), and one line of status text.
Beyond the lookahead there is nothing.

The game loop hands the view one `Frame` per rendered frame (`view.py`): the
Observation, the car pose, the held action, status/result text and, in `--debug`
only, the full road. Any object with `draw(frame)` and `close()` can be the view,
so a 3D renderer can replace `renderer.py: TopDownRenderer` without touching the
game loop: `play(env, controller, view=MyView(), ...)`. A normal view must draw
the road from `frame.obs` only.

Rendering is decoupled from physics: physics always runs at `physics_hz` (60 Hz)
and decisions at `decision_hz` (4 Hz); the screen is drawn at up to `--fps`
frames per second, interpolating the car pose between physics ticks. The frame
rate never changes the simulation or the results.

## Architecture

```
blinddrive/
  config.py        every tunable number + easy/normal/hard presets
  actions.py       Steering (7) x Throttle (5) = 35 Actions, strict validation
  road.py          seeded road generation, projection, sampling (env-internal)
  vehicle.py       kinematic bicycle model with inertia and a grip limit
  observation.py   Observation dataclass + builder (the only thing controllers see)
  env.py           BlindDriveEnv: decision schedule, physics ticks, edge walls, finish/timeout
  controllers/
    base.py        Controller protocol: reset(), act(observation) -> Action
    human.py       keyboard -> Action
    replay.py      replays a logged action sequence
    jev.py         Jev: Observation -> JSON state -> one 35-way choice question -> Action
  typesafe.py      dependency-free TypeSafe System One API client (+ .env loader)
  baselines.py     reference controllers: crude (score 0), reference, oracle (score 100)
  suite.py         CLI: certified-solvable seed suites with anchor times
  scoring.py       effective time, normalised score, scorecard statistics
  bench.py         CLI: headless evaluation of jev or a baseline on a suite
  report.py        CLI: self-contained HTML report comparing runs
  recorder.py      JSONL episode logs + deterministic replay verification
  view.py          Frame (everything a view may draw) + View interface (draw/close)
  renderer.py      minimal top-down pygame View (normal mode draws from the Observation only)
  runner.py        CLI entry point
  replay.py        CLI: verify logs replay exactly
```

### Controller interface

```python
class Controller(Protocol):
    def reset(self) -> None: ...
    def act(self, observation: Observation) -> Action: ...
```

The loop (identical in the GUI and headless runners):

```python
env.reset(); controller.reset()
while not env.done:
    if env.decision_due:
        env.apply_action(controller.act(env.observe()))
    env.tick()      # one physics step at physics_hz
```

`env.apply_action` raises if no decision is due; `env.tick` raises if a decision is
due but missing. The schedule is enforced by the environment, not the runner.

### Observation

| field | meaning |
|-------|---------|
| `speed` | m/s |
| `steering_angle` | actual wheel angle, rad, + = left |
| `heading_error` | car heading − road heading, rad, + = car points left of the road |
| `lateral_offset` | car centre from the centerline, m, + = left |
| `visible_centerline` | points every 1 m in **car-local** coords (x forward, y left) |
| `visible_arclengths` | arc length of each point relative to the car (−view_behind … +lookahead) |
| `road_width`, `lookahead` | constants of the current preset |
| `previous_action` | the action currently being held |
| `elapsed_time`, `distance_travelled`, `distance_to_finish` | progress |
| `touching_edge` | the car is scraping the road edge (wall mode) |

The Observation contains only floats, tuples and the previous Action; no reference
to the road or the environment.

### Actions

```
Steering: HARD_LEFT LEFT SLIGHT_LEFT STRAIGHT SLIGHT_RIGHT RIGHT HARD_RIGHT
Throttle: HARD_BRAKE BRAKE COAST ACCELERATE FULL_ACCELERATE
```

Steering levels set a *target* wheel angle (±100 %, ±50 %, ±1/6 of 30°); the wheels
move toward it at 60°/s. Throttle levels set the tyre acceleration
(−8, −4, 0, +2, +4 m/s²), on top of rolling resistance and drag.

### Physics

Kinematic bicycle model at 60 Hz (`vehicle.py`):

```
dx/dt = v cos θ,   dy/dt = v sin θ,   dθ/dt = v κ,   κ = tan δ / L
dδ/dt limited to max_steering_rate
dv/dt = a_tyre − rolling − drag·v²,   0 ≤ v ≤ max_speed
```

Grip: lateral acceleration `v²κ` is capped at `sqrt(grip² − a_tyre²)`. Above that
the car understeers (turns less than commanded). Braking hard therefore leaves less
grip for turning: brake before the corner, not in it.

### Road edge

The car footprint is a 4.2 × 1.8 m rectangle. It touches the edge when any corner is
farther than `road_width / 2` from the centerline. What happens then is set by
`sim.edge` (or `--edge`):

- **`wall`** (default): the edge is a barrier. The car is moved back just inside the
  road (position only), each new impact keeps 30 % of the speed, and while scraping
  the car cannot go faster than 4 m/s. The run continues. There is also a wall just
  behind the start line. Heading, steering and throttle are never touched: turning
  away from the wall is the controller's job. The Observation has `touching_edge`,
  the result counts `edge_hits` and `edge_contact_time`.
- **`crash`**: the episode ends immediately (strict benchmark mode).

With the default numbers, a "never brake, bounce off the walls" driver finishes
roughly 10–15 % slower than one that brakes for corners.

### Road generation

`generate_road(seed, cfg)` builds a random curvature profile out of straights,
gentle curves and sharp curves (radius + turn angle), smooths it twice with a moving
average (so curvature ramps up and down, no kinks), and integrates it into a
centerline. Turning away from the start direction gets progressively less likely,
which keeps the road from spiralling; candidates that come too close to
themselves are rejected and regenerated from the same RNG stream. Same seed and
config ⇒ bit-identical road.

## Logs and replay

Every finished episode writes `runs/<time>_<preset>_seed<seed>_<controller>.jsonl`:

```
{"type": "header", "seed": ..., "preset": ..., "controller": ..., "config": {vehicle, road, sim}}
{"type": "decision", "time": 3.25, "position": [x, y], "heading": ..., "speed": ...,
 "steering_angle": ..., "observation": {...}, "action": {"steering": "LEFT", "throttle": "BRAKE"}}
...
{"type": "result", "success": ..., "termination": ..., "completion_time": ..., ...}
```

The simulation is deterministic, so seed + config + actions reproduce the episode:

```bash
uv run python -m blinddrive.replay runs/*.jsonl          # re-simulate and check every state
uv run python -m blinddrive.runner --replay runs/X.jsonl  # watch it
```

## Driving with Jev

[Jev](https://typesafe.ai) is TypeSafe AI's System One model: it answers typed
questions about a state with calibrated probabilities. BlindDrive asks it one
*choice* question every decision tick, `action`, whose 35 labels are all legal
actions written `STEERING+THROTTLE` (e.g. `LEFT+BRAKE`). The chosen label is
parsed strictly back into the Action, so steering and throttle are always decided
together.

### Setup (once)

```bash
cd blinddrive
cp .env.example .env        # then put your key in .env:  TYPESAFE_API_KEY=...
uv run python -m blinddrive.bench --check      # verifies the key, lists models
```

### Run

```bash
uv run python -m blinddrive.runner --controller jev --seed 42           # watch Jev drive
uv run python -m blinddrive.bench --suite suites/normal-dev.json --workers 4   # score Jev (20 seeds)
uv run python -m blinddrive.bench --suite suites/hard-dev.json --delay-ms 250  # with a 250 ms reaction delay
uv run python -m blinddrive.bench --dry-run --seed 42                   # print a request, send nothing
```

See [Evaluation](#evaluation) for what the numbers mean.

### What Jev sees and decides

- **State** (`controllers/jev.py: observation_to_state`): the Observation re-expressed
  in plain terms — speed, steering angle, heading vs road, distance to each edge,
  the visible road sampled every 5 m (position in the car frame, direction, curve
  radius), progress, previous action — plus the public rules (limits, decision
  interval, edge rule). It is a pure function of the Observation: no seed, no road,
  nothing beyond the lookahead. `--dry-run` shows exactly what is sent.
- **Question** (`build_questions`): each of the 35 labels' description states its
  exact effect (target wheel angle, turning radius, acceleration in m/s²).
- **No fallback.** If the API fails after retries (2 retries with backoff for
  timeouts, 408/429/5xx) or returns an unknown label, the episode is aborted and
  reported as an error. There is never a default or heuristic action.
- **Latency does not count.** Simulated time is paused while waiting for Jev
  (the GUI keeps rendering and shows "Thinking…"). Real latency is logged.
- **Cost.** One API call per decision: 4 per simulated second, so roughly 120–200
  calls per 500 m episode. Hopeless runs end early as "stalled" (less than 5 m of progress in 20 s).

### Logs

Each decision line additionally carries `controller_info`: model, request id,
latency, attempts, token usage, the answer with its confidence and full 35-way probability
distribution, and the exact `state` sent. The header stores the questions and
rules. `python -m blinddrive.replay` verifies Jev logs like any other (including
aborted ones, up to the point of failure).

## Evaluation

The goal is numbers you can trust and compare: every seed is proven solvable,
every score is anchored to reference controllers on the same seed, and latency,
safety and cost are reported next to the score, not folded into it.

### Reference controllers (`baselines.py`)

| name | what it is | role |
|------|-----------|------|
| `crude` | look 10 m ahead, hold 12 m/s | **score 0** |
| `reference` | pure pursuit + a speed limit assuming the tightest legal curve may start just past the visible range; observation-only | proves solvability; typical score 75–90 |
| oracle | the same planner with the whole road visible (privileged, anchor only) | **score 100** |

They are measuring sticks only; nothing ever falls back to them. The oracle is a
strong planner, not a proven optimum, so scores above 100 are possible.

### Suites (`suite.py`)

A suite is a list of seeds that the observation-only `reference` controller
finishes without touching an edge, i.e. solvable with exactly the information a
controller receives, together with the three anchor times for each seed and a
fingerprint of the rules. `suites/{easy,normal,hard}-dev.json` (20 seeds each)
are committed for development. For a real evaluation, build a private suite with
secret seeds so nothing can be tuned or memorised against it:

```bash
uv run python -m blinddrive.suite build --preset hard --count 30 --random --out private/hard-test.json
uv run python -m blinddrive.suite show suites/normal-dev.json
```

Loading a suite checks its fingerprint and re-runs one seed; if physics or
baselines changed since it was built, it refuses to run ("rebuild the suite").

### Scoring (`scoring.py`)

- **Effective time**: elapsed time, plus distance not driven at 2 m/s if the run
  did not finish (crash, stall, timeout, or controller error).
- **Score per episode** = 100 × (T<sub>crude</sub> − T<sub>eff</sub>) / (T<sub>crude</sub> − T<sub>oracle</sub>),
  floored at −100. Normalising per seed makes easy and hard roads count equally.
- **Run score** = mean over episodes, with a 95% confidence interval.
- Reported separately: finish rate, crashes / stalls / timeouts / errors, edge hits
  and wall-contact time per episode, API latency p50/p95/max, decisions, tokens.
- A run that gains less than 5 m in 20 s ends as **stalled** (this saves API calls
  on hopeless runs and applies to every controller).

### Latency

Simulated time pauses while a remote controller thinks, so the default score is
pure decision quality; the real API latency is still measured and reported.
`--delay-ms N` adds a fixed, reproducible reaction delay: each action takes effect
N ms after the observation it was based on (the previous action keeps running
meanwhile, and Jev is told about the delay in its rules). Scores under delay use
the zero-delay anchors, so the drop shows what latency costs. For reference, the
`reference` controller on `hard-dev` scores about 75 without delay and about −45
with 250 ms.

### Reports (`report.py`)

Every bench run writes `runs/bench-*.json` and a self-contained `runs/bench-*.html`.
Put several runs side by side (scorecard + seed-by-seed table):

```bash
uv run python -m blinddrive.bench --suite suites/hard-dev.json --controller reference
uv run python -m blinddrive.bench --suite suites/hard-dev.json --controller crude
uv run python -m blinddrive.bench --suite suites/hard-dev.json               # jev
uv run python -m blinddrive.report runs/bench-*.json -o runs/compare.html
```

The Jev harness (state format + question wording) carries a version
(`jev-harness-v1`) that is stored in every log and run; scores from different
harness versions are not directly comparable.

## Adding a controller

Implement `reset()` and `act(observation) -> Action`, and run it with
`run_episode(BlindDriveEnv(config, seed), controller)`. It must not receive the
environment, the road, or anything else besides the Observation.
