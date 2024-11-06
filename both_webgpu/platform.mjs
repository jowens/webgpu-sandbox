function platform(navigator) {
  if (typeof process !== "undefined") {
    if (navigator.userAgent?.startsWith("Deno")) {
      return navigator.userAgent;
    } else {
      return process.release.name;
    }
  } else {
    return navigator.userAgent;
  }
}
export { platform };
