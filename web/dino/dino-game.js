/* The Dino game and its glomerular encoding - a line-for-line port of flylab/dino.py.
 *
 * This file deliberately has no p5 dependency. p5 draws the game; it must never run it,
 * because p5 in global mode replaces window.random and a single accidental call would break
 * the parity check silently. Keeping the rules here means Node can run them headless, which
 * is exactly what verify.js does.
 *
 * Every number below is from the Processing original. The two deliberate changes are that
 * time is counted in frames rather than milliseconds, and that the runner holds one fixed
 * x position instead of a random one per agent.
 */

(function (global) {
  "use strict";

  const RUN = 0;
  const JUMP = 1;
  const DUCK = 2;
  const ACTIONS = 3;

  const GROUND_Y = 450;
  const DUCK_Y = 484;
  const STAND_WIDTH = 80;
  const STAND_HEIGHT = 86;
  const DUCK_WIDTH = 110;
  const DUCK_HEIGHT = 52;
  const RUNNER_X = 200;

  const JUMP_HEIGHT = 172;
  const JUMP_START = 0.0001;
  const JUMP_STEP = 0.03;

  const SPAWN_X = 1350;
  const SPEED_0 = 15.0;
  const SPEED_GAIN = 0.001;
  const SPAWN_MIN_FRAMES = 30;
  const SPAWN_MAX_FRAMES = 90;

  const CACTUS_WIDTHS = [30, 64, 98, 46, 96, 146];
  const CACTUS_HEIGHTS = [66, 66, 66, 96, 96, 96];
  const CACTUS_Y = [470, 470, 470, 444, 444, 444];
  const BIRD_WIDTH = 84;
  const BIRD_HEIGHT = 40;
  const BIRD_Y = [435, 480, 370];

  const GAP_MAX = 500.0;
  const GLOMERULI = 55;
  const TUNING_PER_FEATURE = 11;
  const FEATURE_RANGES = [
    [0.0, GAP_MAX],
    [370.0, 480.0],
    [40.0, 96.0],
    [30.0, 146.0],
    [15.0, 20.0],
  ];

  /* numpy's linspace computes start + i*step and then pins the last value to stop exactly.
   * Recomputing it any other way moves the centres by an ulp, which is enough to reorder
   * Kenyon cells sitting on the winner boundary. */
  const CENTRES = FEATURE_RANGES.map(function (range) {
    const step = (range[1] - range[0]) / (TUNING_PER_FEATURE - 1);
    const row = [];
    for (let i = 0; i < TUNING_PER_FEATURE; i++) row.push(range[0] + i * step);
    row[TUNING_PER_FEATURE - 1] = range[1];
    return row;
  });
  const WIDTHS = FEATURE_RANGES.map(function (range) {
    return (range[1] - range[0]) / (TUNING_PER_FEATURE - 1);
  });

  function mulberry32(seed) {
    let a = seed >>> 0;
    return function () {
      a |= 0;
      a = (a + 0x6d2b79f5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function overlaps(a, b) {
    return a.x + a.width > b.x && a.x < b.x + b.width && a.y + a.height > b.y && a.y < b.y + b.height;
  }

  /* Python builds this at float64 and narrows to float32 before the matmul. The narrowing is
   * not cosmetic: it has to happen on both sides or the drive differs in its last bits. */
  function toGlomeruli(features) {
    const activation = new Float32Array(GLOMERULI);
    let at = 0;
    for (let f = 0; f < FEATURE_RANGES.length; f++) {
      for (let i = 0; i < TUNING_PER_FEATURE; i++) {
        const z = (features[f] - CENTRES[f][i]) / WIDTHS[f];
        activation[at++] = Math.exp(-0.5 * z * z);
      }
    }
    return activation;
  }

  function Game(seed) {
    this.random = mulberry32(seed);
    this.speed = SPEED_0;
    this.frame = 0;
    this.y = GROUND_Y;
    this.ducking = false;
    this.jumpStage = 0.0;
    this.obstacles = [];
    this.alive = true;
    this.spawnTimer = this.interval();
  }

  Game.prototype.interval = function () {
    return SPAWN_MIN_FRAMES + Math.trunc(this.random() * (SPAWN_MAX_FRAMES - SPAWN_MIN_FRAMES));
  };

  Game.prototype.jumping = function () {
    return this.jumpStage > 0.0;
  };

  Game.prototype.box = function () {
    if (this.ducking) return { x: RUNNER_X, y: this.y, width: DUCK_WIDTH, height: DUCK_HEIGHT };
    return { x: RUNNER_X, y: this.y, width: STAND_WIDTH, height: STAND_HEIGHT };
  };

  Game.prototype.step = function (action) {
    if (!this.alive) return { cleared: false, crashed: false };

    if (!this.jumping()) this.act(action);
    if (this.jumping()) this.advanceJump();

    const speed = Math.trunc(this.speed);
    const cleared = this.advanceObstacles(speed);
    this.maybeSpawn();

    const self = this.box();
    let crashed = false;
    for (let i = 0; i < this.obstacles.length; i++) {
      if (overlaps(self, this.obstacles[i])) crashed = true;
    }
    if (crashed) this.alive = false;

    this.speed += SPEED_GAIN;
    this.frame += 1;
    return { cleared: cleared, crashed: crashed };
  };

  Game.prototype.act = function (action) {
    if (action === JUMP) {
      this.ducking = false;
      this.y = GROUND_Y;
      this.jumpStage = JUMP_START;
    } else if (action === DUCK) {
      this.ducking = true;
      this.y = DUCK_Y;
    } else {
      this.ducking = false;
      this.y = GROUND_Y;
    }
  };

  Game.prototype.advanceJump = function () {
    const stage = this.jumpStage;
    this.y = Math.trunc(GROUND_Y - -4 * stage * (stage - 1) * JUMP_HEIGHT);
    this.jumpStage += JUMP_STEP;
    if (this.jumpStage > 1.0) {
      this.jumpStage = 0.0;
      this.y = GROUND_Y;
    }
  };

  Game.prototype.advanceObstacles = function (speed) {
    let cleared = false;
    const kept = [];
    for (let i = 0; i < this.obstacles.length; i++) {
      const before = this.obstacles[i];
      /* Spread rather than rebuild. An earlier version listed the four fields the rules use and
       * so quietly dropped `kind` and `bird`, which only the renderer reads - the game and the
       * parity check went on agreeing with Python while the page threw on the second frame.
       * Python moves obstacles with dataclasses.replace, which never had the problem. */
      const after = { ...before, x: before.x - speed };
      if (before.x + before.width >= RUNNER_X && after.x + after.width < RUNNER_X) cleared = true;
      if (after.x + after.width >= 0) kept.push(after);
    }
    this.obstacles = kept;
    return cleared;
  };

  Game.prototype.maybeSpawn = function () {
    this.spawnTimer -= 1;
    if (this.spawnTimer > 0) return;
    this.obstacles.push(this.random() < 0.5 ? this.cactus() : this.bird());
    this.spawnTimer = this.interval();
  };

  Game.prototype.cactus = function () {
    const kind = Math.trunc(this.random() * CACTUS_WIDTHS.length);
    return { x: SPAWN_X, y: CACTUS_Y[kind], width: CACTUS_WIDTHS[kind], height: CACTUS_HEIGHTS[kind], kind: kind, bird: false };
  };

  Game.prototype.bird = function () {
    const kind = Math.trunc(this.random() * BIRD_Y.length);
    return { x: SPAWN_X, y: BIRD_Y[kind], width: BIRD_WIDTH, height: BIRD_HEIGHT, kind: kind, bird: true };
  };

  Game.prototype.nextObstacle = function () {
    for (let i = 0; i < this.obstacles.length; i++) {
      if (this.obstacles[i].x + this.obstacles[i].width >= RUNNER_X) return this.obstacles[i];
    }
    return null;
  };

  Game.prototype.features = function () {
    const ahead = this.nextObstacle();
    if (ahead === null) return [GAP_MAX, 0.0, 0.0, 0.0, this.speed];
    const gap = Math.min(Math.max(ahead.x - RUNNER_X, 0.0), GAP_MAX);
    return [gap, ahead.y, ahead.height, ahead.width, this.speed];
  };

  Game.prototype.glomeruli = function () {
    return toGlomeruli(this.features());
  };

  const DinoGame = {
    RUN: RUN,
    JUMP: JUMP,
    DUCK: DUCK,
    ACTIONS: ACTIONS,
    GROUND_Y: GROUND_Y,
    DUCK_Y: DUCK_Y,
    RUNNER_X: RUNNER_X,
    STAND_WIDTH: STAND_WIDTH,
    STAND_HEIGHT: STAND_HEIGHT,
    DUCK_WIDTH: DUCK_WIDTH,
    DUCK_HEIGHT: DUCK_HEIGHT,
    GAP_MAX: GAP_MAX,
    GLOMERULI: GLOMERULI,
    Game: Game,
    mulberry32: mulberry32,
    overlaps: overlaps,
    toGlomeruli: toGlomeruli,
  };

  if (typeof module !== "undefined" && module.exports) module.exports = DinoGame;
  else global.DinoGame = DinoGame;
})(typeof globalThis !== "undefined" ? globalThis : this);
