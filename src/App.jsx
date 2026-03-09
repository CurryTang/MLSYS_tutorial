import { useEffect, useMemo, useRef, useState } from 'react';
import ReactMarkdown from 'react-markdown';
import rehypeKatex from 'rehype-katex';
import rehypeRaw from 'rehype-raw';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import 'katex/dist/katex.min.css';
import './App.css';

import 'katex/dist/katex.min.css';
import './App.css';

const markdownModules = import.meta.glob('../notes/Mlsys/*.md', {
  eager: true,
  import: 'default',
  query: '?url',
});

const tutorialDefinitions = [
  createTutorialDefinition('MLSYS1', 'MLSYS1.md', 'MLSYS1.en.md'),
  createTutorialDefinition('MLSYS2', 'MLSYS2.md', 'MLSYS2.en.md'),
  createTutorialDefinition('MLSYS3', 'MLSYS3.md', 'MLSYS3.en.md'),
  createTutorialDefinition('MLSYS4', 'MLSYS4.md', 'MLSYS4.en.md'),
  createTutorialDefinition('MLSYS5', 'MLSYS5.md', 'MLSYS5.en.md'),
  createTutorialDefinition('MLSYS6', 'MLSYS6.md', 'MLSYS6.en.md'),
  createTutorialDefinition(
    'MLSYS7 Compute-Bound Kernel (1)',
    'MLSYS7 Compute-Bound Kernel (1).md',
    'MLSYS7 Compute-Bound Kernel (1).en.md',
  ),
  createTutorialDefinition(
    'MLSYS8 Compute-Bound Kernel (2)',
    'MLSYS8 Compute-Bound Kernel (2).md',
    'MLSYS8 Compute-Bound Kernel (2).en.md',
  ),
  createTutorialDefinition(
    'MLSYS9 Compute-bound kernel (3)',
    'MLSYS9 Compute-bound kernel (3).md',
    'MLSYS9 Compute-bound kernel (3).en.md',
  ),
  createTutorialDefinition('MLSYS10 parallelism', 'MLSYS10 parallelism.md', 'MLSYS10 parallelism.en.md'),
  createTutorialDefinition('MLSYS11 nano-vllm-1', 'MLSYS11 nano-vllm-1.md', 'MLSYS11 nano-vllm-1.en.md'),
  createTutorialDefinition('MLSYS12 nano-vllm-2', 'MLSYS12 nano-vllm-2.md', 'MLSYS12 nano-vllm-2.en.md'),
  createTutorialDefinition(
    'MLSYS13 Quantization and precision',
    'MLSYS13 Quantization and precision.md',
    'MLSYS13 Quantization and precision.en.md',
  ),
];

const tutorials = tutorialDefinitions.map((definition) => ({
  ...definition,
  variants: {
    zh: createVariant(definition.zhFileName),
    en: createVariant(definition.enFileName),
  },
}));

const noteIdByAlias = buildNoteAliasMap(tutorials);
const mediaModules = import.meta.glob('../notes/Mlsys/assets/**/*.{png,jpg,jpeg,gif,webp,svg,avif,bmp}', {
  eager: true,
  import: 'default',
  query: '?url',
});
const mediaUrlByAlias = buildMediaAliasMap(mediaModules);
const languageOptions = [
  { id: 'zh', label: '中文' },
  { id: 'en', label: 'English' },
];

function createTutorialDefinition(title, zhFileName, enFileName) {
  return {
    id: zhFileName,
    title,
    fileName: zhFileName,
    zhFileName,
    enFileName,
  };
}

function createVariant(fileName) {
  return {
    fileName,
    url: resolveMarkdownUrl(`../notes/Mlsys/${fileName}`),
  };
}

function resolveMarkdownUrl(modulePath) {
  const url = markdownModules[modulePath];
  if (typeof url !== 'string') {
    throw new Error(`Missing markdown module for ${modulePath}`);
  }
  return url;
}

function normalizePathToken(rawValue) {
  if (!rawValue) {
    return '';
  }

  let value = rawValue.trim().replace(/\\/g, '/');
  try {
    value = decodeURIComponent(value);
  } catch {
    // Ignore malformed URI fragments and keep the original token.
  }

  value = value.replace(/^\.\//, '');
  value = value.replace(/^\//, '');
  value = value.replace(/^notes\//i, '');

  return value.toLowerCase();
}

function buildNoteAliasMap(tutorialList) {
  const map = new Map();

  const addAlias = (alias, id) => {
    const normalized = normalizePathToken(alias);
    if (normalized && !map.has(normalized)) {
      map.set(normalized, id);
    }
  };

  tutorialList.forEach((tutorial) => {
    const fileNames = [tutorial.variants.zh.fileName, tutorial.variants.en.fileName];

    addAlias(tutorial.id, tutorial.id);
    addAlias(tutorial.fileName, tutorial.id);
    addAlias(`Mlsys/${tutorial.fileName}`, tutorial.id);
    addAlias(`notes/Mlsys/${tutorial.fileName}`, tutorial.id);

    fileNames.forEach((fileName) => {
      const withoutMd = fileName.replace(/\.md$/i, '');
      const withoutLang = withoutMd.replace(/\.en$/i, '');
      addAlias(fileName, tutorial.id);
      addAlias(`Mlsys/${fileName}`, tutorial.id);
      addAlias(`notes/Mlsys/${fileName}`, tutorial.id);
      addAlias(withoutMd, tutorial.id);
      addAlias(withoutLang, tutorial.id);
    });
  });

  return map;
}

function buildMediaAliasMap(modules) {
  const map = new Map();

  const addAlias = (alias, url) => {
    const normalized = normalizePathToken(alias);
    if (normalized && !map.has(normalized)) {
      map.set(normalized, url);
    }
  };

  Object.entries(modules).forEach(([modulePath, assetUrl]) => {
    if (typeof assetUrl !== 'string') {
      return;
    }

    const relativePath = modulePath.replace('../notes/', '');
    const fileName = relativePath.split('/').at(-1) ?? relativePath;
    addAlias(relativePath, assetUrl);
    addAlias(`notes/${relativePath}`, assetUrl);
    addAlias(fileName, assetUrl);
    addAlias(`assets/${fileName}`, assetUrl);
    addAlias(`./assets/${fileName}`, assetUrl);
  });

  return map;
}

function splitObsidianTarget(rawContent) {
  const [targetPart, ...aliasParts] = rawContent.split('|');
  const target = targetPart?.trim() ?? '';
  const aliasRaw = aliasParts.join('|').trim();

  if (!aliasRaw || /^\d+$/.test(aliasRaw)) {
    return { target, alias: '' };
  }

  return { target, alias: aliasRaw };
}

function prettyLabel(rawTarget) {
  const [withoutAnchor] = rawTarget.split('#');
  const token = withoutAnchor.split('/').at(-1) ?? withoutAnchor;
  return token.replace(/\.en\.md$/i, '').replace(/\.md$/i, '').trim() || rawTarget.trim();
}

function resolveNoteId(rawTarget) {
  const [withoutAnchor] = rawTarget.split('#');
  const normalized = normalizePathToken(withoutAnchor);

  if (!normalized) {
    return null;
  }

  const basename = normalized.split('/').at(-1) ?? normalized;
  const candidates = [
    normalized,
    normalized.endsWith('.md') ? normalized.slice(0, -3) : `${normalized}.md`,
    basename,
    basename.endsWith('.md') ? basename.slice(0, -3) : `${basename}.md`,
    `mlsys/${basename}`,
    `mlsys/${basename.endsWith('.md') ? basename.slice(0, -3) : `${basename}.md`}`,
  ];

  for (const candidate of candidates) {
    const match = noteIdByAlias.get(candidate);
    if (match) {
      return match;
    }
  }

  return null;
}

function resolveMediaUrl(rawTarget) {
  const [withoutAnchor] = rawTarget.split('#');
  const normalized = normalizePathToken(withoutAnchor);

  if (!normalized) {
    return null;
  }

  const basename = normalized.split('/').at(-1) ?? normalized;
  const candidates = [normalized, basename, `mlsys/assets/${basename}`, `assets/${basename}`];

  for (const candidate of candidates) {
    const match = mediaUrlByAlias.get(candidate);
    if (match) {
      return match;
    }
  }

  return null;
}

function normalizeObsidianMarkdown(markdownText) {
  if (!markdownText) {
    return '';
  }

  let normalized = markdownText;

  normalized = normalized.replace(/%%[\s\S]*?%%/g, '');

  normalized = normalized.replace(/^>\s*\[!([^\]\n+-]+)(?:[+-])?\](.*)$/gim, (_, type, rawTitle) => {
    const label = type.trim();
    const title = rawTitle.trim().replace(/^[-:\s]+/, '');
    const heading = title || (label.charAt(0).toUpperCase() + label.slice(1).toLowerCase());
    return `> **${heading}:**`;
  });

  normalized = normalized.replace(/!\[\[([^\]\n]+)\]\]/g, (_, body) => {
    const { target, alias } = splitObsidianTarget(body);
    if (!target) {
      return '';
    }

    const mediaUrl = resolveMediaUrl(target);
    if (mediaUrl) {
      return `![${alias || prettyLabel(target)}](${mediaUrl})`;
    }

    const noteId = resolveNoteId(target);
    if (noteId) {
      return `[Embedded note: ${alias || prettyLabel(target)}](#${encodeURIComponent(noteId)})`;
    }

    return `*Embedded asset not found: ${alias || prettyLabel(target)}*`;
  });

  normalized = normalized.replace(/\[\[([^\]\n]+)\]\]/g, (_, body) => {
    const { target, alias } = splitObsidianTarget(body);
    if (!target) {
      return '';
    }

    const noteId = resolveNoteId(target);
    if (noteId) {
      return `[${alias || prettyLabel(target)}](#${encodeURIComponent(noteId)})`;
    }

    if (/^https?:\/\//i.test(target)) {
      return `[${alias || target}](${target})`;
    }

    return alias || prettyLabel(target);
  });

  normalized = normalized.replace(/==([^=\n][^=\n]*?)==/g, '<mark>$1</mark>');

  return normalized;
}

function App() {
  const initialHash = decodeURIComponent(window.location.hash.replace(/^#/, ''));
  const initialId = tutorials.find((tutorial) => tutorial.id === initialHash)?.id ?? tutorials[0]?.id ?? '';

  const [selectedTutorialId, setSelectedTutorialId] = useState(initialId);
  const [language, setLanguage] = useState('zh');
  const [query, setQuery] = useState('');
  const [contentByKey, setContentByKey] = useState({});
  const [errorByKey, setErrorByKey] = useState({});
  const inFlightRef = useRef(new Set());

  const filteredTutorials = useMemo(() => {
    const normalizedQuery = query.trim().toLowerCase();
    if (!normalizedQuery) {
      return tutorials;
    }

    return tutorials.filter((tutorial) =>
      [tutorial.title, tutorial.fileName].some((field) => field.toLowerCase().includes(normalizedQuery)),
    );
  }, [query]);

  const selectedTutorial =
    tutorials.find((tutorial) => tutorial.id === selectedTutorialId) ?? filteredTutorials[0] ?? tutorials[0] ?? null;

  const activeLanguage = selectedTutorial?.variants[language] ? language : 'zh';
  const selectedVariant = selectedTutorial?.variants[activeLanguage] ?? null;
  const contentKey = selectedTutorial && selectedVariant ? `${selectedTutorial.id}:${activeLanguage}` : '';

  useEffect(() => {
    if (!selectedTutorial) {
      return;
    }

    const encoded = `#${encodeURIComponent(selectedTutorial.id)}`;
    if (window.location.hash !== encoded) {
      window.history.replaceState(null, '', encoded);
    }
  }, [selectedTutorial]);

  useEffect(() => {
    const handleHashChange = () => {
      const hashValue = decodeURIComponent(window.location.hash.replace(/^#/, ''));
      if (tutorials.some((tutorial) => tutorial.id === hashValue)) {
        setSelectedTutorialId(hashValue);
      }
    };

    window.addEventListener('hashchange', handleHashChange);
    return () => window.removeEventListener('hashchange', handleHashChange);
  }, []);

  useEffect(() => {
    if (!selectedVariant || !contentKey) {
      return;
    }

    const isLoaded = Object.prototype.hasOwnProperty.call(contentByKey, contentKey);
    if (isLoaded || errorByKey[contentKey] || inFlightRef.current.has(contentKey)) {
      return;
    }

    inFlightRef.current.add(contentKey);

    fetch(selectedVariant.url)
      .then((response) => {
        if (!response.ok) {
          throw new Error(`Unable to load markdown (${response.status})`);
        }
        return response.text();
      })
      .then((content) => {
        setContentByKey((prev) => ({
          ...prev,
          [contentKey]: content,
        }));
      })
      .catch((error) => {
        setErrorByKey((prev) => ({
          ...prev,
          [contentKey]: error.message,
        }));
      })
      .finally(() => {
        inFlightRef.current.delete(contentKey);
      });
  }, [contentByKey, contentKey, errorByKey, selectedVariant]);

  const hasSelectedContent = contentKey
    ? Object.prototype.hasOwnProperty.call(contentByKey, contentKey)
    : false;
  const selectedContent = hasSelectedContent ? contentByKey[contentKey] : '';
  const selectedError = contentKey ? errorByKey[contentKey] : '';
  const selectedIsLoading = Boolean(selectedTutorial && selectedVariant && !hasSelectedContent && !selectedError);

  const normalizedSelectedContent = useMemo(
    () => normalizeObsidianMarkdown(selectedContent),
    [selectedContent],
  );

  return (
    <div className="app-shell">
      <aside className="notes-panel">
        <header className="panel-header">
          <p className="eyebrow">ML Systems Tutorial</p>
          <h1>Reading Room</h1>
          <p className="panel-meta">{tutorials.length} published tutorials</p>
        </header>

        <label className="search">
          <span>Search Tutorials</span>
          <input
            type="search"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Type tutorial name or filename"
          />
        </label>

        <div className="note-list">
          {filteredTutorials.map((tutorial) => (
            <button
              key={tutorial.id}
              className={`note-button ${selectedTutorial?.id === tutorial.id ? 'active' : ''}`}
              onClick={() => setSelectedTutorialId(tutorial.id)}
              type="button"
            >
              <span className="note-title">{tutorial.title}</span>
              <span className="note-subtitle">{tutorial.fileName}</span>
            </button>
          ))}
          {filteredTutorials.length === 0 && (
            <p className="list-empty">No tutorials matched your search.</p>
          )}
        </div>
      </aside>

      <main className="reader-panel">
        {selectedTutorial ? (
          <>
            <header className="reader-header">
              <div className="reader-header-top">
                <div>
                  <p className="reader-label">Current Tutorial</p>
                  <h2>{selectedTutorial.title}</h2>
                  <p>{selectedVariant?.fileName ?? selectedTutorial.fileName}</p>
                </div>

                <div className="language-toggle" aria-label="Language selector" role="group">
                  {languageOptions.map((option) => (
                    <button
                      key={option.id}
                      className={`language-button ${activeLanguage === option.id ? 'active' : ''}`}
                      onClick={() => setLanguage(option.id)}
                      type="button"
                      aria-pressed={activeLanguage === option.id}
                    >
                      {option.label}
                    </button>
                  ))}
                </div>
              </div>
            </header>

            <article className="markdown-body">
              {selectedError && <p className="empty-note">Load failed: {selectedError}</p>}
              {selectedIsLoading && !selectedError && <p className="empty-note">Loading markdown...</p>}
              {!selectedIsLoading && !selectedError && normalizedSelectedContent?.trim() && (
                <ReactMarkdown
                  remarkPlugins={[remarkGfm, remarkMath]}
                  rehypePlugins={[rehypeRaw, rehypeKatex]}
                  components={{
                    a: ({ href, children, ...props }) => {
                      const external = href?.startsWith('http');
                      return (
                        <a
                          href={href}
                          target={external ? '_blank' : undefined}
                          rel={external ? 'noreferrer' : undefined}
                          {...props}
                        >
                          {children}
                        </a>
                      );
                    },
                  }}
                >
                  {normalizedSelectedContent}
                </ReactMarkdown>
              )}
              {!selectedIsLoading && !selectedError && selectedContent !== undefined && !selectedContent.trim() && (
                <p className="empty-note">This file is empty and ready for future notes.</p>
              )}
            </article>
          </>
        ) : (
          <section className="reader-empty">
            <h2>No published Markdown files found</h2>
            <p>Add ready notes to the published allowlist and refresh.</p>
          </section>
        )}
      </main>
    </div>
  );
}

export default App;
