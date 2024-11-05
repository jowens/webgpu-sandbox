"use strict";

import { main, getDeviceAndAdapter } from "./both.mjs";
import {
  Canvas,
  EventType,
  PixelFormat,
  Rect,
  Texture,
  TextureAccess,
  Window,
  WindowBuilder,
} from "../../../src/deno_sdl2/mod.ts";

class MinimumWindow {
  dimensions = {
    width: 800,
    height: 800,
  };
  screenDimensions = {
    width: 800,
    height: 800,
  };
  texture;
  canvas;
  window;
  constructor(device) {
    const window = new WindowBuilder(
      "Hi, Deno Minimum Window!",
      this.dimensions.width,
      this.dimensions.height
    ).build();
    this.canvas = window.canvas();
    this.window = window;
    const creator = this.canvas.textureCreator();
    this.sdl2texture = creator.createTexture(
      PixelFormat.ABGR8888,
      TextureAccess.Streaming,
      this.dimensions.width,
      this.dimensions.height
    );
    this.texture = this.device.createTexture({
      label: "Capture",
      size: this.dimensions,
      format: "rgba8unorm-srgb",
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
    });
    const { padded } = getRowPadding(this.dimensions.width);
    this.outputBuffer = this.device.createBuffer({
      label: "Capture",
      size: padded * this.dimensions.height,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
  }
}

const { adapter, device } = await getDeviceAndAdapter(navigator);
// const minwin = new MinimumWindow(device);
const window = new WindowBuilder("Hello, Deno!", 640, 480).build();
const canvas = window.canvas();

async function loop(frames) {
  if (frames == 0) {
    return;
  }
  const event = (await window.events().next()).value;
  if (event.type == EventType.Quit) {
    return;
  } else {
    // if (event.type == EventType.Draw) {
    // Rainbow effect
    const r = Math.sin(Date.now() / 1000) * 127 + 128;
    const g = Math.sin(Date.now() / 1000 + 2) * 127 + 128;
    const b = Math.sin(Date.now() / 1000 + 4) * 127 + 128;
    canvas.setDrawColor(Math.floor(r), Math.floor(g), Math.floor(b), 255);
    canvas.clear();
    await main(navigator);
    canvas.present();
  }
  await loop(frames - 1);
}

async function loopx(frames) {
  if (frames == 0) {
    return;
  }
  await main(navigator);
  minwin.canvas.present();
  await loopx(frames - 1);
}

await loop(2);
