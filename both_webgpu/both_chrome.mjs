import { main } from "http://localhost:8000/webgpu-sandbox/both_webgpu/both.mjs";
if (typeof process !== "undefined" && process.release.name === "node") {
  // running in Node
  alert("Use this only from a web browser.");
} else {
  // running in browser
}

main(navigator);
