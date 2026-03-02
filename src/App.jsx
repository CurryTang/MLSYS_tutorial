import { useEffect, useMemo, useRef, useState } from 'react';
import ReactMarkdown from 'react-markdown';
import rehypeKatex from 'rehype-katex';
import rehypeRaw from 'rehype-raw';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import note1Url from '../notes/Mlsys/MLSYS1.md?url';
import note2Url from '../notes/Mlsys/MLSYS2.md?url';
import note3Url from '../notes/Mlsys/MLSYS3.md?url';
import note4Url from '../notes/Mlsys/MLSYS4.md?url';
import note5Url from '../notes/Mlsys/MLSYS5.md?url';
import note6Url from '../notes/Mlsys/MLSYS6.md?url';
import note7Url from '../notes/Mlsys/MLSYS7 Compute-Bound Kernel (1).md?url';
import note8Url from '../notes/Mlsys/MLSYS8 Compute-Bound Kernel (2).md?url';
import note9Url from '../notes/Mlsys/MLSYS9 Compute-bound kernel (3).md?url';
import note10Url from '../notes/Mlsys/MLSYS10 parallelism.md?url';
import note11Url from '../notes/Mlsys/MLSYS11 nano-vllm-1.md?url';
import 'katex/dist/katex.min.css';
import './App.css';

const notes = [
  createNote('Mlsys/MLSYS1.md', note1Url),
  createNote('Mlsys/MLSYS2.md', note2Url),
  createNote('Mlsys/MLSYS3.md', note3Url),
  createNote('Mlsys/MLSYS4.md', note4Url),
  createNote('Mlsys/MLSYS5.md', note5Url),
  createNote('Mlsys/MLSYS6.md', note6Url),
  createNote('Mlsys/MLSYS7 Compute-Bound Kernel (1).md', note7Url),
  createNote('Mlsys/MLSYS8 Compute-Bound Kernel (2).md', note8Url),
  createNote('Mlsys/MLSYS9 Compute-bound kernel (3).md', note9Url),
  createNote('Mlsys/MLSYS10 parallelism.md', note10Url),
  createNote('Mlsys/MLSYS11 nano-vllm-1.md', note11Url),
];

const noteIdByAlias = buildNoteAliasMap(notes);
const mediaModules = import.meta.glob('../notes/Mlsys/assets/**/*.{png,jpg,jpeg,gif,webp,svg,avif,bmp}', {
  eager: true,
  import: 'default',
  query: '?url',
});
const mediaUrlByAlias = buildMediaAliasMap(mediaModules);

function createNote(id, url) {
  const fileName = id.split('/').at(-1) ?? id;
  return {
    id,
    fileName,
    title: fileName.replace(/\.md$/i, ''),
    url,
  };
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

function buildNoteAliasMap(noteList) {
  const map = new Map();

  const addAlias = (alias, id) => {
    const normalized = normalizePathToken(alias);
    if (normalized && !map.has(normalized)) {
      map.set(normalized, id);
    }
  };

  noteList.forEach((note) => {
    const base = note.fileName.replace(/\.md$/i, '');
    addAlias(note.id, note.id);
    addAlias(note.fileName, note.id);
    addAlias(base, note.id);
    addAlias(`Mlsys/${note.fileName}`, note.id);
    addAlias(`notes/Mlsys/${note.fileName}`, note.id);
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
  return token.replace(/\.md$/i, '').trim() || rawTarget.trim();
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

  // Remove Obsidian comments.
  normalized = normalized.replace(/%%[\s\S]*?%%/g, '');

  // Convert Obsidian callouts into standard Markdown blockquotes.
  normalized = normalized.replace(/^>\s*\[!([^\]\n+-]+)(?:[+-])?\](.*)$/gim, (_, type, rawTitle) => {
    const label = type.trim();
    const title = rawTitle.trim().replace(/^[-:\s]+/, '');
    const heading = title || (label.charAt(0).toUpperCase() + label.slice(1).toLowerCase());
    return `> **${heading}:**`;
  });

  // Convert embedded links and media: ![[...]].
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

  // Convert wiki-links: [[target|alias]].
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

  // Obsidian highlight syntax.
  normalized = normalized.replace(/==([^=\n][^=\n]*?)==/g, '<mark>$1</mark>');

  return normalized;
}

function App() {
  const initialHash = decodeURIComponent(window.location.hash.replace(/^#/, ''));
  const initialId = notes.find((note) => note.id === initialHash)?.id ?? notes[0]?.id ?? '';

  const [selectedId, setSelectedId] = useState(initialId);
  const [query, setQuery] = useState('');
  const [contentById, setContentById] = useState({});
  const [errorById, setErrorById] = useState({});
  const inFlightRef = useRef(new Set());

  const filteredNotes = useMemo(() => {
    const normalizedQuery = query.trim().toLowerCase();
    if (!normalizedQuery) {
      return notes;
    }

    return notes.filter((note) =>
      [note.title, note.fileName].some((field) => field.toLowerCase().includes(normalizedQuery)),
    );
  }, [query]);

  const selectedNote =
    notes.find((note) => note.id === selectedId) ?? filteredNotes[0] ?? notes[0] ?? null;

  useEffect(() => {
    if (!selectedNote) {
      return;
    }

    const encoded = `#${encodeURIComponent(selectedNote.id)}`;
    if (window.location.hash !== encoded) {
      window.history.replaceState(null, '', encoded);
    }
  }, [selectedNote]);

  useEffect(() => {
    const handleHashChange = () => {
      const hashValue = decodeURIComponent(window.location.hash.replace(/^#/, ''));
      if (notes.some((note) => note.id === hashValue)) {
        setSelectedId(hashValue);
      }
    };

    window.addEventListener('hashchange', handleHashChange);
    return () => window.removeEventListener('hashchange', handleHashChange);
  }, []);

  useEffect(() => {
    if (!selectedNote || !selectedNote.url) {
      return;
    }

    const isLoaded = Object.prototype.hasOwnProperty.call(contentById, selectedNote.id);
    if (isLoaded || errorById[selectedNote.id] || inFlightRef.current.has(selectedNote.id)) {
      return;
    }

    inFlightRef.current.add(selectedNote.id);

    fetch(selectedNote.url)
      .then((response) => {
        if (!response.ok) {
          throw new Error(`Unable to load markdown (${response.status})`);
        }
        return response.text();
      })
      .then((content) => {
        setContentById((prev) => ({
          ...prev,
          [selectedNote.id]: content,
        }));
      })
      .catch((error) => {
        setErrorById((prev) => ({
          ...prev,
          [selectedNote.id]: error.message,
        }));
      })
      .finally(() => {
        inFlightRef.current.delete(selectedNote.id);
      });
  }, [contentById, errorById, selectedNote]);

  const hasSelectedContent = selectedNote
    ? Object.prototype.hasOwnProperty.call(contentById, selectedNote.id)
    : false;
  const selectedContent = selectedNote && hasSelectedContent ? contentById[selectedNote.id] : '';
  const selectedError = selectedNote ? errorById[selectedNote.id] : '';
  const selectedIsLoading = Boolean(selectedNote && !hasSelectedContent && !selectedError);

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
          <p className="panel-meta">{notes.length} published notes</p>
        </header>

        <label className="search">
          <span>Search Notes</span>
          <input
            type="search"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Type filename or keyword"
          />
        </label>

        <div className="note-list">
          {filteredNotes.map((note) => (
            <button
              key={note.id}
              className={`note-button ${selectedNote?.id === note.id ? 'active' : ''}`}
              onClick={() => setSelectedId(note.id)}
              type="button"
            >
              <span className="note-title">{note.title}</span>
              <span className="note-subtitle">{note.fileName}</span>
            </button>
          ))}
          {filteredNotes.length === 0 && (
            <p className="list-empty">No notes matched your search.</p>
          )}
        </div>
      </aside>

      <main className="reader-panel">
        {selectedNote ? (
          <>
            <header className="reader-header">
              <p className="reader-label">Current Document</p>
              <h2>{selectedNote.title}</h2>
              <p>{selectedNote.fileName}</p>
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
