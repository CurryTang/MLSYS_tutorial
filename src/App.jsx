import { useEffect, useMemo, useRef, useState } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import './App.css';

const noteModules = import.meta.glob('../notes/**/*.md', {
  eager: true,
  import: 'default',
  query: '?url',
});

const notes = Object.entries(noteModules)
  .map(([modulePath, assetUrl]) => {
    const relativePath = modulePath.replace('../notes/', '');
    const fileName = relativePath.split('/').at(-1) ?? relativePath;
    return {
      id: relativePath,
      fileName,
      title: fileName.replace(/\.md$/i, ''),
      url: typeof assetUrl === 'string' ? assetUrl : '',
    };
  })
  .sort((a, b) =>
    a.fileName.localeCompare(b.fileName, undefined, {
      numeric: true,
      sensitivity: 'base',
    }),
  );

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

  return (
    <div className="app-shell">
      <aside className="notes-panel">
        <header className="panel-header">
          <p className="eyebrow">ML Systems Tutorial</p>
          <h1>Reading Room</h1>
          <p className="panel-meta">{notes.length} Markdown files synced locally</p>
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
              {!selectedIsLoading && !selectedError && selectedContent?.trim() && (
                <ReactMarkdown
                  remarkPlugins={[remarkGfm]}
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
                  {selectedContent}
                </ReactMarkdown>
              )}
              {!selectedIsLoading && !selectedError && selectedContent !== undefined && !selectedContent.trim() && (
                <p className="empty-note">This file is empty and ready for future notes.</p>
              )}
            </article>
          </>
        ) : (
          <section className="reader-empty">
            <h2>No Markdown files found</h2>
            <p>Add `.md` files to `notes/` and refresh.</p>
          </section>
        )}
      </main>
    </div>
  );
}

export default App;
