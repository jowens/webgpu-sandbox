"use strict";

// Loading a .node file into a recent version of Node is ... challenging
// https://stackoverflow.com/questions/77913169/loading-native-node-addons-from-es-module
import { createRequire } from "node:module";
const require = createRequire(import.meta.url);
const dawn = require("../../../src/dawn-build/dawn.node");
Object.assign(globalThis, dawn.globals); // Provides constants like GPUBufferUsage.MAP_READ

import * as Plot from "@observablehq/plot";
import { JSDOM } from "jsdom";

let navigator = {
  gpu: dawn.create(["enable-dawn-features=use_user_defined_labels_in_backend"]),
};

import { main } from "./both.mjs";
if (typeof process !== "undefined" && process.release.name === "node") {
  // running in Node
} else {
  // running in browser
  alert("Use this only in Node.");
}

const data = await main(navigator);
console.log(data);
const plot = Plot.plot({
  document: new JSDOM("").window.document,
  marks: [
    Plot.lineY(data, {
      x: "bytesTransferred",
      y: "bandwidth",
      stroke: "workgroupSize",
    }),
  ],
});

plot.setAttributeNS(
  "http://www.w3.org/2000/xmlns/",
  "xmlns",
  "http://www.w3.org/2000/svg"
);
plot.setAttributeNS(
  "http://www.w3.org/2000/xmlns/",
  "xmlns:xlink",
  "http://www.w3.org/1999/xlink"
);

process.stdout.write(plot.outerHTML);
