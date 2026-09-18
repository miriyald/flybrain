/* Drawing only. p5 renders; dino-game.js decides what there is to render.
 *
 * p5 runs in instance mode on purpose. In global mode it replaces window.random, and a single
 * accidental call from game logic would desynchronise this page from the Python it was trained
 * in, silently and in a way the parity check could not see. Instance mode makes that mistake
 * impossible rather than merely discouraged.
 *
 * The game advances on a fixed 60 Hz accumulator rather than once per rendered frame. p5 draws
 * at the display's refresh rate, so on a 120 Hz monitor an unpinned loop would run the game at
 * twice the speed it was trained at and roughly double its difficulty.
 *
 * There is deliberately no preload(). p5's preload blocks setup until every asset resolves, and
 * a page opened from the filesystem cannot fetch anything - which left this page sitting on
 * "Loading..." forever with a blank canvas. The sprite sheet now arrives as a data URI from
 * sprites.js and is loaded after setup, so the game starts immediately and draws plain hitboxes
 * until the art is ready. If the art never arrives the demo still runs.
 */

(function () {
  "use strict";

  const STEP_MS = 1000 / 60;
  const MAX_CATCHUP_MS = 250;

  const VIEW_TOP = 262;
  const SCALE = 0.75;
  const CANVAS_W = 1280 * SCALE;
  const CANVAS_H = 232;

  const SPRITES = {
    standing: [1338, 2, 88, 94],
    walk1: [1514, 2, 88, 94],
    walk2: [1602, 2, 88, 94],
    crouch1: [1866, 36, 118, 60],
    crouch2: [1984, 36, 118, 60],
    cactus: [
      [446, 2, 34, 70],
      [480, 2, 68, 70],
      [548, 2, 102, 70],
      [652, 2, 50, 100],
      [702, 2, 100, 100],
      [802, 2, 150, 100],
    ],
    bird1: [260, 2, 92, 80],
    bird2: [352, 2, 92, 80],
    ground: [2, 104, 2400, 24],
  };

  const CELL_COLS = 64;

  const state = {
    game: null,
    model: null,
    decision: null,
    seed: 1,
    runs: 0,
    best: 0,
    cleared: 0,
    paused: true,
    started: false,
    hitboxes: false,
    groundOffset: 0,
    sheet: null,
  };

  function screenY(worldY) {
    return (worldY - VIEW_TOP) * SCALE;
  }

  /* One frame. A run that ends stops the clock rather than rolling straight into another one,
   * so the crash stays on screen to be looked at. The next run starts when it is asked for. */
  function advance() {
    if (!state.game.alive) {
      state.paused = true;
      return;
    }
    const decision = FlyDino.decide(state.game, state.model);
    if (!decision.committed) state.decision = decision;
    const step = state.game.step(decision.action);
    if (step.cleared) state.cleared += 1;
    state.best = Math.max(state.best, state.game.frame);
    state.groundOffset = (state.groundOffset + Math.trunc(state.game.speed)) % 2400;

    if (!state.game.alive) {
      state.runs += 1;
      state.paused = true;
      showOutcome();
      updateControls();
    }
  }

  function showOutcome() {
    const notice = document.getElementById("outcome");
    const cleared = state.cleared;
    notice.textContent =
      `Crashed after ${state.game.frame} frames, having cleared ${cleared} obstacle${cleared === 1 ? "" : "s"}.`;
    notice.hidden = false;
  }

  function startRun(nextSeed) {
    state.seed = nextSeed;
    state.game = new DinoGame.Game(state.seed);
    state.cleared = 0;
    state.decision = null;
    state.started = true;
    state.paused = false;
    document.getElementById("outcome").hidden = true;
    paintReadout();
    updateControls();
  }

  /* Three states, and the buttons should only ever offer what makes sense in each: stopped
   * (Play, stepping allowed), running (Pause, stepping meaningless), crashed (Run again,
   * nothing left to step). */
  function updateControls() {
    const play = document.getElementById("pause");
    const running = state.game.alive && !state.paused;
    if (!state.game.alive) play.textContent = "Run again";
    else if (running) play.textContent = "Pause";
    else play.textContent = state.started ? "Resume" : "Play";
    play.classList.toggle("primary", !running);
    document.getElementById("step").disabled = !state.game.alive || running;
  }

  function drawSprite(p, slice, x, y) {
    p.image(state.sheet, x * SCALE, screenY(y), slice[2] * SCALE, slice[3] * SCALE, slice[0], slice[1], slice[2], slice[3]);
  }

  function drawBox(p, box) {
    p.rect(box.x * SCALE, screenY(box.y), box.width * SCALE, box.height * SCALE);
  }

  function render(p) {
    p.clear();
    const game = state.game;

    if (state.sheet) {
      const shift = -state.groundOffset;
      drawSprite(p, SPRITES.ground, shift, 515);
      drawSprite(p, SPRITES.ground, shift + 2400, 515);
    } else {
      p.noStroke();
      p.fill(150, 160, 170);
      p.rect(0, screenY(515), CANVAS_W, 2);
    }

    for (const obstacle of game.obstacles) {
      if (state.sheet) {
        if (obstacle.bird) {
          const frame = Math.trunc(game.frame / 15) % 2 === 0 ? SPRITES.bird1 : SPRITES.bird2;
          drawSprite(p, frame, obstacle.x - 4, obstacle.y - 16);
        } else {
          drawSprite(p, SPRITES.cactus[obstacle.kind], obstacle.x - 2, obstacle.y - 2);
        }
      } else {
        p.noStroke();
        p.fill(obstacle.bird ? "#9a6216" : "#4b6b52");
        drawBox(p, obstacle);
      }
    }

    if (state.sheet) {
      let dino;
      if (game.jumping()) dino = SPRITES.standing;
      else if (game.ducking) dino = Math.trunc(game.frame / 6) % 2 === 0 ? SPRITES.crouch1 : SPRITES.crouch2;
      else dino = Math.trunc(game.frame / 6) % 2 === 0 ? SPRITES.walk1 : SPRITES.walk2;
      drawSprite(p, dino, DinoGame.RUNNER_X - 4, game.y - 2);
    } else {
      p.noStroke();
      p.fill("#1f8f56");
      drawBox(p, game.box());
    }

    if (state.hitboxes) {
      p.noFill();
      p.strokeWeight(1.5);
      p.stroke(224, 92, 92);
      for (const obstacle of game.obstacles) drawBox(p, obstacle);
      p.stroke(46, 160, 100);
      drawBox(p, game.box());
      p.noStroke();
    }
  }

  /* Once per rendered frame, not once per game step - the accumulator can run several steps
   * in a row after a stall, and the panels only need the state the viewer is about to see. */
  function paintReadout() {
    document.getElementById("frames").textContent = String(state.game.frame);
    document.getElementById("runs").textContent = String(state.runs);
    document.getElementById("best").textContent = String(state.best);
    document.getElementById("cleared").textContent = String(state.cleared);
    paintBars();
    paintCells();
  }

  /* Bars show preference, not magnitude.
   *
   * Both readouts start at one and only ever shrink, so after training every score is usually
   * negative and the *least* wanted action has the largest absolute value. Sizing bars by |score|
   * therefore drew the longest bar next to the action the circuit least wanted - exactly
   * backwards. Rescaling between the worst and best score on screen means the fullest bar is
   * always the one being taken. The raw number is still printed beside it. */
  const FLOOR = 6;

  function paintBars() {
    const decision = state.decision;
    if (!decision || !decision.scores) return;
    const scores = decision.scores;
    let low = Infinity;
    let high = -Infinity;
    for (let a = 0; a < scores.length; a++) {
      low = Math.min(low, scores[a]);
      high = Math.max(high, scores[a]);
    }
    const span = Math.max(high - low, 1e-6);

    for (let a = 0; a < scores.length; a++) {
      const share = (scores[a] - low) / span;
      const bar = document.getElementById("bar" + a);
      bar.style.width = (FLOOR + (100 - FLOOR) * share).toFixed(1) + "%";
      bar.className = a === decision.action ? "bar chosen" : "bar";
      document.getElementById("val" + a).textContent = scores[a].toFixed(1);
    }
    const doing = state.game.jumping() ? "mid-jump - committed" : ["running", "jumping", "ducking"][decision.action];
    document.getElementById("doing").textContent = doing;
  }

  function paintCells() {
    const canvas = document.getElementById("cells");
    if (!state.decision || !state.decision.code) return;
    const context = canvas.getContext("2d");
    const code = state.decision.code;
    const rows = Math.ceil(code.length / CELL_COLS);
    const wide = canvas.width / CELL_COLS;
    const tall = canvas.height / rows;
    const style = getComputedStyle(document.body);
    const live = style.getPropertyValue("--fire").trim();
    const dim = style.getPropertyValue("--line").trim();

    context.clearRect(0, 0, canvas.width, canvas.height);
    for (let i = 0; i < code.length; i++) {
      context.fillStyle = code[i] ? live : dim;
      context.fillRect((i % CELL_COLS) * wide + 1, Math.trunc(i / CELL_COLS) * tall + 1, wide - 2, tall - 2);
    }
  }

  const sketch = function (p) {
    let accumulator = 0;

    p.setup = function () {
      p.createCanvas(CANVAS_W, CANVAS_H);
      p.noSmooth();
      p.noStroke();

      state.model = window.FLYDINO_MODEL || null;
      if (!state.model) {
        document.getElementById("missing").hidden = false;
        p.noLoop();
        return;
      }
      state.game = new DinoGame.Game(state.seed);
      /* Show the opening position's code straight away. Nothing has moved yet, but the circuit
       * already has an opinion about an empty track, and an empty grey panel suggests otherwise. */
      state.decision = FlyDino.decide(state.game, state.model);
      paintReadout();
      updateControls();

      if (window.FLYDINO_SPRITES) {
        p.loadImage(
          window.FLYDINO_SPRITES,
          function (loaded) {
            state.sheet = loaded;
          },
          function () {
            document.getElementById("plain").hidden = false;
          }
        );
      } else {
        document.getElementById("plain").hidden = false;
      }
    };

    p.draw = function () {
      if (!state.game) return;
      if (!state.paused) {
        accumulator += Math.min(p.deltaTime, MAX_CATCHUP_MS);
        while (accumulator >= STEP_MS) {
          advance();
          accumulator -= STEP_MS;
        }
      }
      render(p);
      paintReadout();
    };

    p.keyPressed = function () {
      if (p.key === "h" || p.key === "H") state.hitboxes = !state.hitboxes;
      if (p.key === " ") {
        togglePlay();
        return false;
      }
      return true;
    };
  };

  function togglePlay() {
    if (!state.game) return;
    if (!state.game.alive) startRun(state.seed + 1);
    else {
      state.started = true;
      state.paused = !state.paused;
      document.getElementById("outcome").hidden = true;
    }
    updateControls();
  }

  window.addEventListener("DOMContentLoaded", function () {
    new p5(sketch, document.getElementById("stage"));

    document.getElementById("pause").addEventListener("click", togglePlay);

    document.getElementById("step").addEventListener("click", function () {
      if (!state.game || !state.game.alive) return;
      state.started = true;
      state.paused = true;
      advance();
      paintReadout();
      updateControls();
    });

    document.getElementById("toggle").addEventListener("click", function () {
      state.hitboxes = !state.hitboxes;
      this.textContent = state.hitboxes ? "Hide hitboxes" : "Show hitboxes";
    });
  });
})();
