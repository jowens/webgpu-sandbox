// begin NODE
import { createRequire } from "node:module";
const require = createRequire(import.meta.url);
const dawn = require("../../src/dawn-build/dawn.node");
Object.assign(globalThis, dawn.globals); // Provides constants like GPUBufferUsage.MAP_READ

let navigator = {
  gpu: dawn.create(["enable-dawn-features=use_user_defined_labels_in_backend"]),
};
// end NODE

import { main } from "./node-deno-timing-mre.mjs";

main(navigator);
