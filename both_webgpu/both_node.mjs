"use strict";

// Loading a .node file into a recent version of Node is ... challenging
// https://stackoverflow.com/questions/77913169/loading-native-node-addons-from-es-module
import { createRequire } from "module";
const require = createRequire(import.meta.url);
const dawn = require("../../../src/dawn-build/dawn.node");
Object.assign(globalThis, dawn.globals); // Provides constants like GPUBufferUsage.MAP_READ

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

main(navigator);
