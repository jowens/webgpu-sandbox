"use strict";

import { main } from "./both.mjs";
import { EventType, WindowBuilder } from "../../../src/deno_sdl2/mod.ts";

const window = new WindowBuilder("Hello, Deno!", 640, 480).build();
const canvas = window.canvas();

async function loop(frames, waitingForKeyboard = true) {
  if (frames == 0) {
    return;
  }
  const event = (await window.events().next()).value;
  switch (event.type) {
    case EventType.Quit:
      return;
    case EventType.KeyDown:
      waitingForKeyboard = false;
      break;
    default:
      if (waitingForKeyboard == false) {
        const r = Math.sin(Date.now() / 1000) * 127 + 128;
        const g = Math.sin(Date.now() / 1000 + 2) * 127 + 128;
        const b = Math.sin(Date.now() / 1000 + 4) * 127 + 128;
        canvas.setDrawColor(Math.floor(r), Math.floor(g), Math.floor(b), 255);
        canvas.clear();
        await main(navigator);
        canvas.present();
        frames--;
      }
      break;
  }
  await loop(frames, waitingForKeyboard);
}

await loop(2);
