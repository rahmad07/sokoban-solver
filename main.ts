// Sokoban solver: push-optimal, room by room.
//
// Idea: a level is a hallway with independent rooms off it (one door each). A box
// pushed out of a door into the hallway can never come back, so every box stays in
// its own room and the level's push optimum is the sum of the rooms' optima.
// Each room (3-4 boxes) is solved with a breadth-first search over pushes, which
// makes the first solved state push-optimal. Dead squares (a box there can never
// reach a goal, found by pulling boxes back from the goals) and frozen 2x2 blocks
// are pruned; neither prunes a solvable state, so the optimum stands.
//
// Original code, written for this entry. Deterministic: fixed exploration order,
// no clock, no randomness. Node 22+, standard library only.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const IN_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

// Up, down, left, right: step offset, step letter, push letter.
const DIRS = [
  { dx: 0, dy: -1, step: 'u', push: 'U' },
  { dx: 0, dy: 1, step: 'd', push: 'D' },
  { dx: -1, dy: 0, step: 'l', push: 'L' },
  { dx: 1, dy: 0, step: 'r', push: 'R' },
];

type Board = {
  w: number;
  h: number;
  wall: Uint8Array;
  goals: number[];
  boxes: number[];
  player: number;
};

/** Parse XSB text. Short rows and the outer frame count as walls. */
function parseLevel(text: string): Board {
  const rows = text.split('\n').filter((r) => r.length > 0);
  const h = rows.length;
  const w = Math.max(...rows.map((r) => r.length));
  const wall = new Uint8Array(w * h).fill(1);
  const goals: number[] = [];
  const boxes: number[] = [];
  let player = -1;
  rows.forEach((row, y) => {
    for (let x = 0; x < row.length; x++) {
      const ch = row[x];
      const i = y * w + x;
      if (ch === '#') continue;
      wall[i] = 0;
      if (ch === '.' || ch === '*' || ch === '+') goals.push(i);
      if (ch === '$' || ch === '*') boxes.push(i);
      if (ch === '@' || ch === '+') player = i;
    }
  });
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (x === 0 || y === 0 || x === w - 1 || y === h - 1) wall[y * w + x] = 1;
    }
  }
  return { w, h, wall, goals, boxes: boxes.sort((a, b) => a - b), player };
}

/**
 * Live squares: a square is live when a box standing there can still be pushed
 * onto some goal. Found by pulling boxes back from the goals: if a box on x can
 * reach a goal, a box on y = x - d can be pushed to x when the player can stand
 * on y - d. Squares that never become live are dead: a box there is lost.
 */
function liveSquares(b: Board): Uint8Array {
  const live = new Uint8Array(b.w * b.h);
  const queue: number[] = [];
  for (const g of b.goals) {
    live[g] = 1;
    queue.push(g);
  }
  for (let qi = 0; qi < queue.length; qi++) {
    const x = queue[qi];
    for (const d of DIRS) {
      const off = d.dy * b.w + d.dx;
      const y = x - off;
      if (y < 0 || y >= live.length) continue;
      if (!b.wall[y] && !b.wall[y - off] && !live[y]) {
        live[y] = 1;
        queue.push(y);
      }
    }
  }
  return live;
}

/**
 * Flood fill from `from` over open squares avoiding `occupied` boxes. Marks the
 * region in `seen` with `stamp` and returns its lowest square (the region's name).
 */
function floodRegion(
  b: Board,
  occupied: Uint8Array,
  from: number,
  seen: Int32Array,
  stamp: number,
  stack: Int32Array,
): number {
  let low = from;
  let n = 0;
  stack[n++] = from;
  seen[from] = stamp;
  while (n > 0) {
    const p = stack[--n];
    if (p < low) low = p;
    for (const d of DIRS) {
      const q = p + d.dy * b.w + d.dx;
      if (q < 0 || q >= seen.length) continue;
      if (!b.wall[q] && !occupied[q] && seen[q] !== stamp) {
        seen[q] = stamp;
        stack[n++] = q;
      }
    }
  }
  return low;
}

/** True when the box just pushed to `t` sits in a frozen 2x2 block. */
function frozenBlock(
  b: Board,
  occupied: Uint8Array,
  isGoal: Uint8Array,
  t: number,
): boolean {
  const w = b.w;
  for (const a of [t, t - 1, t - w, t - w - 1]) {
    if (a < 0 || a + w + 1 >= occupied.length) continue;
    const cells = [a, a + 1, a + w, a + w + 1];
    let allBlocked = true;
    let offGoal = false;
    for (const c of cells) {
      if (!b.wall[c] && !occupied[c]) {
        allBlocked = false;
        break;
      }
      if (occupied[c] && !isGoal[c]) offGoal = true;
    }
    if (allBlocked && offGoal) return true;
  }
  return false;
}

/** Shortest walk from `from` to `to` around walls and boxes, in step letters. */
function walkTo(b: Board, occupied: Uint8Array, from: number, to: number): string {
  if (from === to) return '';
  const prev = new Int32Array(b.w * b.h).fill(-1);
  const how: string[] = new Array(b.w * b.h);
  const queue = [from];
  prev[from] = from;
  for (let qi = 0; qi < queue.length; qi++) {
    const p = queue[qi];
    if (p === to) break;
    for (const d of DIRS) {
      const q = p + d.dy * b.w + d.dx;
      if (q < 0 || q >= prev.length) continue;
      if (b.wall[q] || occupied[q] || prev[q] !== -1) continue;
      prev[q] = p;
      how[q] = d.step;
      queue.push(q);
    }
  }
  if (prev[to] === -1) throw new Error('walkTo: push square out of reach');
  let s = '';
  for (let p = to; p !== from; p = prev[p]) s = how[p] + s;
  return s;
}

type Solution = { pushes: number; moves: string; player: number };

/**
 * Push-optimal solution of one board by breadth-first search over pushes. A
 * state is the boxes plus the player's reachable region; states are expanded in
 * order of pushes made, so the first solved state met is optimal. Returns null
 * when the search gives up (state cap) or the board cannot be solved.
 */
function searchPushOptimal(board: Board, maxStates: number): Solution | null {
  const { w, wall, goals } = board;
  const size = wall.length;
  const isGoal = new Uint8Array(size);
  for (const g of goals) isGoal[g] = 1;
  const live = liveSquares(board);
  const solved = (boxes: number[]): boolean => boxes.every((b) => isGoal[b]);
  if (board.boxes.some((b) => !live[b])) return null;

  const occupied = new Uint8Array(size);
  const seen = new Int32Array(size);
  const mark = new Int32Array(size);
  const stack = new Int32Array(size);
  let stamp = 0;

  // State key: the region's lowest square folded with the boxes' squares.
  const keyOf = (boxes: number[], low: number): number => {
    let k = low;
    for (const b of boxes) k = k * 256 + b;
    return k * 8 + boxes.length;
  };

  const boxesOf: number[][] = [board.boxes.slice()];
  const playerOf: number[] = [board.player];
  const parent: number[] = [-1];
  const pushFrom: number[] = [-1];
  const pushDir: number[] = [-1];
  const index = new Map<number, number>();
  for (const b of board.boxes) occupied[b] = 1;
  const startLow = floodRegion(board, occupied, board.player, seen, ++stamp, stack);
  for (const b of board.boxes) occupied[b] = 0;
  index.set(keyOf(board.boxes, startLow), 0);
  if (solved(board.boxes)) return { pushes: 0, moves: '', player: board.player };

  let found = -1;
  for (let head = 0; found < 0 && head < boxesOf.length; head++) {
    const boxes = boxesOf[head];
    for (const b of boxes) occupied[b] = 1;
    const here = ++stamp;
    floodRegion(board, occupied, playerOf[head], seen, here, stack);
    for (let bi = 0; bi < boxes.length && found < 0; bi++) {
      const b = boxes[bi];
      for (let di = 0; di < DIRS.length; di++) {
        const d = DIRS[di];
        const off = d.dy * w + d.dx;
        const t = b + off;
        // The player must reach the square behind the box; the landing square
        // must be open, empty and live.
        if (seen[b - off] !== here || wall[t] || occupied[t] || !live[t]) continue;
        occupied[b] = 0;
        occupied[t] = 1;
        if (frozenBlock(board, occupied, isGoal, t)) {
          occupied[t] = 0;
          occupied[b] = 1;
          continue;
        }
        const next = boxes.slice();
        next[bi] = t;
        next.sort((a, b2) => a - b2);
        // NOTE: the new state's region goes in `mark`, not `seen`: `seen`
        // still holds this state's region, which the remaining pushes need.
        const low = floodRegion(board, occupied, b, mark, ++stamp, stack);
        const key = keyOf(next, low);
        occupied[t] = 0;
        occupied[b] = 1;
        if (index.has(key)) continue;
        if (boxesOf.length >= maxStates) {
          for (const x of boxes) occupied[x] = 0;
          return null;
        }
        index.set(key, boxesOf.length);
        boxesOf.push(next);
        playerOf.push(b);
        parent.push(head);
        pushFrom.push(b);
        pushDir.push(di);
        if (solved(next)) {
          found = boxesOf.length - 1;
          break;
        }
      }
    }
    for (const x of boxes) occupied[x] = 0;
  }
  if (found < 0) return null;

  // Rebuild the moves: walk to each push, then push.
  const path: number[] = [];
  for (let s = found; s > 0; s = parent[s]) path.push(s);
  path.reverse();
  let moves = '';
  let player = board.player;
  let boxes = board.boxes;
  for (const s of path) {
    const nextBoxes = boxesOf[s];
    const from = boxes.find((b) => !nextBoxes.includes(b))!;
    const to = nextBoxes.find((b) => !boxes.includes(b))!;
    const di = pushDir[s];
    const d = DIRS[di];
    const off = d.dy * w + d.dx;
    for (const b of boxes) occupied[b] = 1;
    moves += walkTo(board, occupied, player, from - off);
    for (const b of boxes) occupied[b] = 0;
    moves += d.push;
    player = from;
    boxes = nextBoxes;
  }
  return { pushes: path.length, moves, player };
}

type Room = { boxes: number[]; goals: number[] };

/**
 * The level's rooms: cut the dead squares of the player's row (the hallway: a
 * box there reaches no goal) out of the floor, and each remaining part holding a
 * box is a room. Null when the level is no set of rooms.
 */
function findRooms(b: Board): Room[] | null {
  const live = liveSquares(b);
  const row = Math.floor(b.player / b.w);
  const cut = new Uint8Array(b.w * b.h);
  for (let x = 0; x < b.w; x++) {
    const i = row * b.w + x;
    if (!b.wall[i] && !live[i]) cut[i] = 1;
  }
  if (b.boxes.some((box) => cut[box])) return null;
  const part = new Int32Array(b.w * b.h).fill(-1);
  const rooms: Room[] = [];
  for (let s = 0; s < b.w * b.h; s++) {
    if (b.wall[s] || cut[s] || part[s] >= 0) continue;
    const squares = [s];
    part[s] = rooms.length;
    for (let i = 0; i < squares.length; i++) {
      const p = squares[i];
      for (const d of DIRS) {
        const q = p + d.dy * b.w + d.dx;
        if (q < 0 || q >= part.length) continue;
        if (!b.wall[q] && !cut[q] && part[q] < 0) {
          part[q] = rooms.length;
          squares.push(q);
        }
      }
    }
    rooms.push({ boxes: [], goals: [] });
  }
  for (const box of b.boxes) rooms[part[box]].boxes.push(box);
  for (const g of b.goals) {
    if (part[g] < 0) return null;
    rooms[part[g]].goals.push(g);
  }
  if (rooms.some((r) => r.boxes.length !== r.goals.length)) return null;
  return rooms.filter((r) => r.boxes.length > 0);
}

/** Solve one level room by room; the rooms' optima add up to the level's. */
function solveLevel(board: Board): string {
  const rooms = findRooms(board);
  if (rooms === null) {
    const whole = searchPushOptimal(board, 500_000);
    return whole ? whole.moves : '';
  }
  let player = board.player;
  let moves = '';
  const done: boolean[] = rooms.map(() => false);
  for (let ri = 0; ri < rooms.length; ri++) {
    const room = rooms[ri];
    const wall = board.wall.slice();
    for (let oj = 0; oj < rooms.length; oj++) {
      if (oj === ri) continue;
      for (const sq of done[oj] ? rooms[oj].goals : rooms[oj].boxes) wall[sq] = 1;
    }
    const sub: Board = {
      w: board.w,
      h: board.h,
      wall,
      goals: room.goals.slice().sort((a, b) => a - b),
      boxes: room.boxes.slice().sort((a, b) => a - b),
      player,
    };
    const solved = searchPushOptimal(sub, 1_000_000);
    if (solved === null) {
      const whole = searchPushOptimal(board, 500_000);
      return whole ? whole.moves : '';
    }
    moves += solved.moves;
    player = solved.player;
    done[ri] = true;
  }
  return moves;
}

for (const id of readdirSync(IN_DIR).sort()) {
  const moves = solveLevel(parseLevel(readFileSync(join(IN_DIR, id, 'level.txt'), 'utf8')));
  mkdirSync(join(OUT_DIR, id), { recursive: true });
  writeFileSync(join(OUT_DIR, id, 'solution.txt'), `${moves}\n`);
}
