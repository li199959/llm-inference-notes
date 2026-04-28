const chapters = [
  ["01", "推理问题", "chapter-01-the-inference-problem.md"],
  ["02", "推理硬件基础", "chapter-02-hardware-foundations-for-inference.md"],
  ["03", "Transformer 推理机制", "chapter-03-transformer-inference-mechanics.md"],
  ["04", "量化", "chapter-04-quantization.md"],
  ["05", "推测解码", "chapter-05-speculative-decoding.md"],
  ["06", "KV Cache 优化", "chapter-06-kv-cache-optimization.md"],
  ["07", "Kernel 工程与 FlashAttention", "chapter-07-kernel-engineering-and-flashattention.md"],
  ["08", "服务系统架构", "chapter-08-serving-systems-architecture.md"],
  ["09", "大模型推理并行", "chapter-09-parallelism-for-large-model-inference.md"],
  ["10", "长上下文与内存管理", "chapter-10-long-context-and-memory-management.md"],
  ["11", "边缘与端侧推理", "chapter-11-edge-and-on-device-inference.md"],
  ["12", "推理时计算扩展", "chapter-12-inference-time-compute-scaling.md"],
  ["13", "可观测性与生产工程", "chapter-13-observability-and-production-engineering.md"],
  ["14", "推理的未来", "chapter-14-the-future-of-inference.md"],
  ["A", "关键公式参考", "appendix-a-key-formulas-reference.md"],
  ["B", "论文阅读清单", "appendix-b-annotated-paper-reading-list.md"],
  ["C", "术语表", "appendix-c-glossary.md"],
  ["D", "硬件对比", "appendix-d-hardware-comparison.md"],
].map(([number, title, file], index) => ({
  index,
  number,
  title,
  file,
  path: `./translated_md/${file}`,
  text: "",
  html: "",
}));

const state = {
  active: Number(localStorage.getItem("llm-reader-active") || 0),
  query: "",
  font: Number(localStorage.getItem("llm-reader-font") || 18),
};

const els = {
  chapterList: document.querySelector("#chapterList"),
  content: document.querySelector("#content"),
  outline: document.querySelector("#outline"),
  search: document.querySelector("#searchInput"),
  currentKind: document.querySelector("#currentKind"),
  currentTitle: document.querySelector("#currentTitle"),
  prev: document.querySelector("#prevChapter"),
  next: document.querySelector("#nextChapter"),
  progress: document.querySelector("#progressBar"),
  sidebar: document.querySelector("#sidebar"),
};

document.documentElement.style.setProperty("--reader-font", `${state.font}px`);
document.documentElement.dataset.theme = localStorage.getItem("llm-reader-theme") || "light";

function escapeHtml(value) {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function slugify(value) {
  return value
    .toLowerCase()
    .replace(/[^\p{L}\p{N}]+/gu, "-")
    .replace(/^-+|-+$/g, "");
}

function inlineMarkdown(value) {
  let html = escapeHtml(value);
  html = html.replace(/`([^`]+)`/g, (_, content) => {
    if (isProbablyInlineMath(content)) return formatInlineMath(content);
    return `<code>${content}</code>`;
  });
  html = html.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  html = html.replace(/\*([^*]+)\*/g, "<em>$1</em>");
  return html;
}

function isProbablyInlineMath(value) {
  const text = value.trim();
  if (!text) return false;
  if (/^\\\(.+\\\)$/.test(text) || /^\$.+\$$/.test(text)) return true;
  if (/\\(frac|operatorname|mathrm|mathbb|text|sum|max|min|Delta|hat|alpha|ell|quad|times|sqrt|top|odot|cdot|infty|approx|partial|pi|mid|le|ge|mathbf)/.test(text)) return true;
  if (/[A-Za-z]_[A-Za-z0-9{:-}]+/.test(text)) return true;
  if (/[A-Za-z]\^\{?.+\}?/.test(text)) return true;
  if (/[=|]/.test(text) && /[A-Za-z]/.test(text)) return true;
  return false;
}

function formatInlineMath(value) {
  return `\\(${normalizeInlineMath(value)}\\)`;
}

function normalizeInlineMath(value) {
  let text = value.trim();
  if (/^\\\(.+\\\)$/.test(text)) text = text.slice(2, -2).trim();
  if (/^\$.+\$$/.test(text)) text = text.slice(1, -1).trim();
  text = text.replace(/·/g, "\\cdot ");
  text = text.replace(/×/g, "\\times ");
  text = text.replace(/sqrt\(([^)]+)\)/g, "\\sqrt{$1}");
  text = text.replace(/\b(softmax|round|GeLU|GELU|max|min)\b/g, "\\operatorname{$1}");
  text = text.replace(/\b([A-Za-z])(\d+)\b/g, "$1_{$2}");
  text = text.replace(/_([A-Za-z][A-Za-z0-9]*)\b/g, "_{$1}");
  return text;
}

function isMathBlockLine(value) {
  if (!value || value.startsWith("|") || value.startsWith("#")) return false;
  if (/^\d+\.\s+/.test(value) || /^[-*]\s+/.test(value)) return false;
  if (/^\$.*\$$/.test(value) || /^\\\[.*\\\]$/.test(value)) return true;
  if (/[\u3400-\u9fff]/.test(value)) return false;
  if (/\\(frac|operatorname|mathrm|mathbb|text|sum|max|min|Delta|hat|alpha|ell|quad|times|sqrt|top|odot|cdot|infty|approx)/.test(value)) {
    return value.length <= 220;
  }
  if (/^\d+(\.\d+)?\s+(GB|MB|TB)$/.test(value)) return true;
  if (/^[A-Za-z0-9_\\{}()[\]\s,./+\-^=×≤≥<>:]+$/.test(value) && /[=×^]|O\(|FLOP|GB\/s|bytes|TOPS|TFLOP/.test(value)) {
    return value.length <= 220;
  }
  return false;
}

function unit(value) {
  return `\\mathrm{${value}}`;
}

function replaceMathUnits(value) {
  const replacements = [
    ["GB/s", unit("GB/s")],
    ["TB/s", unit("TB/s")],
    ["FLOP/s", unit("FLOP/s")],
    ["FLOP", unit("FLOP")],
    ["bytes", unit("bytes")],
    ["GB", unit("GB")],
    ["MB", unit("MB")],
    ["TB", unit("TB")],
    ["ms", unit("ms")],
  ];
  let text = value;
  replacements.forEach(([raw], index) => {
    const escaped = raw.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    text = text.replace(new RegExp(`\\b${escaped}\\b`, "g"), `@@UNIT_${index}@@`);
  });
  replacements.forEach(([, latex], index) => {
    text = text.replaceAll(`@@UNIT_${index}@@`, latex);
  });
  return text;
}

function normalizeMath(value) {
  let text = value.trim();
  if (text.startsWith("$") || text.startsWith("\\[")) return text;
  if (text === "Achieved Performance (FLOP/s) = min (Peak FLOP/s, Arithmetic Intensity × Peak Bandwidth)") {
    return "\\[\\begin{aligned}\\text{Achieved Performance} &= \\min(\\\\ &\\quad \\text{Peak FLOP/s},\\\\ &\\quad \\text{Arithmetic Intensity} \\times \\text{Peak Bandwidth})\\end{aligned}\\]";
  }
  if (text === "S = QK^T / sqrt(d_k), P = softmax(S), O = PV") {
    return "\\[\\begin{aligned}S &= \\frac{QK^T}{\\sqrt{d_k}},\\\\ P &= \\operatorname{softmax}(S),\\\\ O &= PV\\end{aligned}\\]";
  }
  text = text.replace(/sqrt\(([^)]+)\)/g, "\\sqrt{$1}");
  text = text.replace(/softmax/g, "\\operatorname{softmax}");
  text = text.replace(/max/g, "\\max");
  text = text.replace(/min/g, "\\min");
  text = text.replace(/Σ/g, "\\sum");
  text = text.replace(/ℓ/g, "\\ell");
  text = replaceMathUnits(text);
  text = text.replace(/\s+(\\mathrm\{)/g, "\\,$1");
  text = text.replace(/×/g, "\\times ");
  text = text.replace(/\^([A-Za-z0-9{}]+)/g, "^{$1}");
  return `\\[${text}\\]`;
}

function renderMathBlock(block) {
  const raw = block.map((line) => line.trim());
  if (raw.length === 2 && /^\d+(\.\d+)?\s+GB$/.test(raw[0]) && /^\d+(\.\d+)?\s+GB\/s\s*=\s*\d+(\.\d+)?\s*ms$/.test(raw[1])) {
    const numerator = raw[0].match(/^(\d+(?:\.\d+)?)\s+GB$/)[1];
    const [, denominator, result] = raw[1].match(/^(\d+(?:\.\d+)?)\s+GB\/s\s*=\s*(\d+(?:\.\d+)?)\s*ms$/);
    return `<div class="math-block">\\[\\frac{${numerator}\\,${unit("GB")}}{${denominator}\\,${unit("GB/s")}} = ${result}\\,${unit("ms")}\\]</div>`;
  }
  return `<div class="math-block">${raw.map(normalizeMath).join("\n")}</div>`;
}

function isProbablyMathFence(language, lines) {
  if (/^(math|latex|tex|katex)$/i.test(language)) return true;
  if (!/^(text|txt|plain|plaintext)$/i.test(language)) return false;
  const raw = lines.map((line) => line.trim()).filter(Boolean);
  if (!raw.length) return false;
  return raw.every((line) => isMathBlockLine(line));
}

function renderTable(lines) {
  const rows = lines.map((line) =>
    line
      .trim()
      .replace(/^\|/, "")
      .replace(/\|$/, "")
      .split("|")
      .map((cell) => inlineMarkdown(cell.trim()))
  );
  const head = rows[0] || [];
  const body = rows.slice(2);
  return `<table><thead><tr>${head.map((cell) => `<th>${cell}</th>`).join("")}</tr></thead><tbody>${body
    .map((row) => `<tr>${row.map((cell) => `<td>${cell}</td>`).join("")}</tr>`)
    .join("")}</tbody></table>`;
}

function markdownToHtml(markdown) {
  const lines = markdown.replace(/\r\n/g, "\n").split("\n");
  const html = [];
  let i = 0;

  while (i < lines.length) {
    const line = lines[i];
    const trimmed = line.trim();

    if (!trimmed) {
      i += 1;
      continue;
    }

    if (
      /^--- 第 \d+ 页 ---$/.test(trimmed) ||
      /^_来源页/.test(trimmed) ||
      /^高效 LLM 推理/.test(trimmed) ||
      /^AI 工程内幕 \d+$/.test(trimmed)
    ) {
      i += 1;
      continue;
    }

    if (trimmed.startsWith("```")) {
      const language = trimmed.replace(/^```/, "").trim();
      const code = [];
      i += 1;
      while (i < lines.length && !lines[i].trim().startsWith("```")) {
        code.push(lines[i]);
        i += 1;
      }
      if (i < lines.length) i += 1;
      if (isProbablyMathFence(language, code)) {
        html.push(renderMathBlock(code.filter((line) => line.trim())));
        continue;
      }
      html.push(`<pre><code class="language-${escapeHtml(language)}">${escapeHtml(code.join("\n"))}</code></pre>`);
      continue;
    }

    const heading = trimmed.match(/^(#{1,4})\s+(.+)$/);
    if (heading) {
      const level = heading[1].length;
      const text = heading[2].replace(/#+$/, "").trim();
      const id = slugify(text) || `section-${html.length}`;
      html.push(`<h${level} id="${id}">${inlineMarkdown(text)}</h${level}>`);
      i += 1;
      continue;
    }

    if (trimmed.startsWith("|") && lines[i + 1]?.trim().startsWith("|")) {
      const tableLines = [];
      while (i < lines.length && lines[i].trim().startsWith("|")) {
        tableLines.push(lines[i]);
        i += 1;
      }
      html.push(renderTable(tableLines));
      continue;
    }

    if (isMathBlockLine(trimmed)) {
      const block = [trimmed];
      i += 1;
      while (i < lines.length && !lines[i].trim()) i += 1;
      while (i < lines.length && isMathBlockLine(lines[i].trim())) {
        block.push(lines[i].trim());
        i += 1;
        while (i < lines.length && !lines[i].trim()) i += 1;
      }
      html.push(renderMathBlock(block));
      continue;
    }

    if (/^[-*]\s+/.test(trimmed)) {
      const items = [];
      while (i < lines.length && /^[-*]\s+/.test(lines[i].trim())) {
        items.push(`<li>${inlineMarkdown(lines[i].trim().replace(/^[-*]\s+/, ""))}</li>`);
        i += 1;
      }
      html.push(`<ul>${items.join("")}</ul>`);
      continue;
    }

    if (/^\d+\.\s+/.test(trimmed)) {
      const items = [];
      while (i < lines.length && /^\d+\.\s+/.test(lines[i].trim())) {
        items.push(`<li>${inlineMarkdown(lines[i].trim().replace(/^\d+\.\s+/, ""))}</li>`);
        i += 1;
      }
      html.push(`<ol>${items.join("")}</ol>`);
      continue;
    }

    if (trimmed.startsWith(">")) {
      const quote = [];
      while (i < lines.length && lines[i].trim().startsWith(">")) {
        quote.push(lines[i].trim().replace(/^>\s?/, ""));
        i += 1;
      }
      html.push(`<blockquote>${inlineMarkdown(quote.join(" "))}</blockquote>`);
      continue;
    }

    const paragraph = [];
    while (
      i < lines.length &&
      lines[i].trim() &&
      !/^(#{1,4})\s+/.test(lines[i].trim()) &&
      !/^[-*]\s+/.test(lines[i].trim()) &&
      !/^\d+\.\s+/.test(lines[i].trim()) &&
      !lines[i].trim().startsWith("|") &&
      !lines[i].trim().startsWith(">")
    ) {
      paragraph.push(lines[i].trim());
      i += 1;
    }
    html.push(`<p>${inlineMarkdown(paragraph.join(" "))}</p>`);
  }

  return html.join("\n");
}

function highlight(html, query) {
  if (!query) return html;
  const escaped = query.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  return html.replace(new RegExp(`(${escaped})`, "gi"), "<mark>$1</mark>");
}

function renderChapterList() {
  const q = state.query.trim().toLowerCase();
  els.chapterList.innerHTML = chapters
    .map((chapter) => {
      const count = q ? (chapter.text.toLowerCase().match(new RegExp(q.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"), "g")) || []).length : 0;
      const hidden = q && count === 0 && !chapter.title.toLowerCase().includes(q);
      return `<button class="chapter-link ${chapter.index === state.active ? "active" : ""}" data-index="${chapter.index}" ${hidden ? "hidden" : ""}>
        <span class="chapter-index">${chapter.number}</span>
        <span class="chapter-name">${chapter.title}</span>
        ${q ? `<span class="chapter-count">${count} 处匹配</span>` : ""}
      </button>`;
    })
    .join("");
}

function renderOutline() {
  const headings = [...els.content.querySelectorAll("h2, h3")];
  els.outline.innerHTML = headings
    .map((heading) => `<a class="outline-link level-${heading.tagName.slice(1)}" href="#${heading.id}">${heading.textContent}</a>`)
    .join("");
}

function setActive(index) {
  state.active = Math.max(0, Math.min(index, chapters.length - 1));
  localStorage.setItem("llm-reader-active", String(state.active));
  const chapter = chapters[state.active];
  els.currentKind.textContent = chapter.number.length === 1 && /[A-D]/.test(chapter.number) ? "Appendix" : "Chapter";
  els.currentTitle.textContent = `${chapter.number}. ${chapter.title}`;
  els.content.innerHTML = highlight(chapter.html, state.query.trim());
  els.prev.disabled = state.active === 0;
  els.next.disabled = state.active === chapters.length - 1;
  renderChapterList();
  renderOutline();
  typesetMath();
  window.scrollTo({ top: 0, behavior: "instant" });
  els.sidebar.classList.remove("open");
}

function typesetMath(attempt = 0) {
  if (window.MathJax?.typesetPromise) {
    window.MathJax.typesetPromise([els.content]).catch(() => {});
  } else if (window.MathJax?.startup?.promise) {
    window.MathJax.startup.promise.then(() => typesetMath()).catch(() => {});
  } else if (attempt < 30) {
    window.setTimeout(() => typesetMath(attempt + 1), 250);
  }
}

async function loadChapters() {
  await Promise.all(
    chapters.map(async (chapter) => {
      const response = await fetch(chapter.path);
      if (!response.ok) throw new Error(`无法读取 ${chapter.file}`);
      const buffer = await response.arrayBuffer();
      chapter.text = new TextDecoder("utf-8").decode(buffer);
      chapter.html = markdownToHtml(chapter.text);
    })
  );
  setActive(state.active);
  if (location.hash) {
    window.setTimeout(() => {
      document.getElementById(decodeURIComponent(location.hash.slice(1)))?.scrollIntoView();
    }, 500);
  }
}

els.chapterList.addEventListener("click", (event) => {
  const button = event.target.closest("[data-index]");
  if (button) setActive(Number(button.dataset.index));
});

els.search.addEventListener("input", (event) => {
  state.query = event.target.value;
  renderChapterList();
  setActive(state.active);
});

els.prev.addEventListener("click", () => setActive(state.active - 1));
els.next.addEventListener("click", () => setActive(state.active + 1));

document.querySelector("#fontDown").addEventListener("click", () => {
  state.font = Math.max(15, state.font - 1);
  localStorage.setItem("llm-reader-font", String(state.font));
  document.documentElement.style.setProperty("--reader-font", `${state.font}px`);
});

document.querySelector("#fontUp").addEventListener("click", () => {
  state.font = Math.min(22, state.font + 1);
  localStorage.setItem("llm-reader-font", String(state.font));
  document.documentElement.style.setProperty("--reader-font", `${state.font}px`);
});

document.querySelector("#themeToggle").addEventListener("click", () => {
  const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
  document.documentElement.dataset.theme = next;
  localStorage.setItem("llm-reader-theme", next);
});

document.querySelector("#openNav").addEventListener("click", () => els.sidebar.classList.add("open"));
document.querySelector("#closeNav").addEventListener("click", () => els.sidebar.classList.remove("open"));

window.addEventListener("scroll", () => {
  const max = document.documentElement.scrollHeight - window.innerHeight;
  const pct = max <= 0 ? 0 : (window.scrollY / max) * 100;
  els.progress.style.width = `${Math.min(100, Math.max(0, pct))}%`;
});

loadChapters().catch((error) => {
  els.content.innerHTML = `<h1>载入失败</h1><p>${escapeHtml(error.message)}</p>`;
});
