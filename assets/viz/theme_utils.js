/**
 * Minimal theme utilities shared by HyperNodes visual frontends.
 * Provides host background + light/dark inference without imposing palettes.
 */
(function (window) {
  const ns = (window.HyperNodesTheme = window.HyperNodesTheme || {});

  function parseColorString(value) {
    if (!value) return null;
    try {
      const scratch = document.createElement("div");
      scratch.style.color = value;
      scratch.style.backgroundColor = value;
      scratch.style.display = "none";
      document.body.appendChild(scratch);
      const resolved = getComputedStyle(scratch).color || "";
      scratch.remove();

      const nums = resolved.match(/[\d\.]+/g);
        if (nums && nums.length >= 3) {
          const [r, g, b] = nums.slice(0, 3).map(Number);
          // Check alpha if present
          if (nums.length >= 4) {
             const alpha = Number(nums[3]);
             if (alpha < 0.1) return null;
          }
          const luminance = 0.299 * r + 0.587 * g + 0.114 * b;
          return { r, g, b, luminance, resolved, raw: value };
        }
    } catch (_) {}
    return null;
  }

  function detectHostTheme() {
    const attempts = [];
    const push = (value, source) => {
      if (value) attempts.push({ value: String(value).trim(), source });
    };

    // Parent document (e.g., VS Code iframe host)
    try {
      const parentDoc = window.parent?.document;
      if (parentDoc) {
        const rootStyle = getComputedStyle(parentDoc.documentElement);
        const bodyStyle = getComputedStyle(parentDoc.body);
        push(rootStyle.getPropertyValue("--vscode-editor-background"), "parent --vscode-editor-background");
        push(bodyStyle.backgroundColor, "parent body background");
        push(rootStyle.backgroundColor, "parent root background");
      }
    } catch (_) {}

    // Current document
    try {
      const rootStyle = getComputedStyle(document.documentElement);
      const bodyStyle = getComputedStyle(document.body);
      push(rootStyle.getPropertyValue("--vscode-editor-background"), "--vscode-editor-background");
      push(bodyStyle.backgroundColor, "body background");
      push(rootStyle.backgroundColor, "root background");
    } catch (_) {}

    // Choose first parsable candidate
    let chosen = attempts.find((c) => parseColorString(c.value));
    if (!chosen && attempts.length) chosen = attempts[0];

    const parsed = chosen ? parseColorString(chosen.value) : null;
    let theme = parsed && typeof parsed.luminance === "number"
      ? parsed.luminance > 150
        ? "light"
        : "dark"
      : null;
    let source = chosen?.source || "fallback";

    // VS Code explicit attributes win
    try {
      const docForAttrs = window.parent?.document || document;
      const themeKind = docForAttrs?.body?.getAttribute("data-vscode-theme-kind") || "";
      const bodyClass = docForAttrs?.body?.className || "";
      if (themeKind.includes("light") || bodyClass.includes("vscode-light")) {
        theme = "light";
        source = "vscode attribute";
      } else if (themeKind.includes("dark") || bodyClass.includes("vscode-dark")) {
        theme = "dark";
        source = "vscode attribute";
      }
    } catch (_) {}

    // Fallback to prefers-color-scheme
    if (!theme && window.matchMedia) {
      if (window.matchMedia("(prefers-color-scheme: light)").matches) {
        theme = "light";
        source = "prefers-color-scheme";
      } else if (window.matchMedia("(prefers-color-scheme: dark)").matches) {
        theme = "dark";
        source = "prefers-color-scheme";
      }
    }

    return {
      theme: theme || "dark",
      background: parsed?.resolved || parsed?.raw || chosen?.value || "transparent",
      luminance: parsed?.luminance ?? null,
      source,
    };
  }

  ns.parseColorString = parseColorString;
  ns.detectHostTheme = detectHostTheme;
})(window);
