import { main } from "./both.mjs";
if (typeof process !== "undefined" && process.release.name === "node") {
  // running in Node
} else {
  // running in browser
  alert("Use this only in Node.");
}

main();
