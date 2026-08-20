CREATE TABLE releases (
    id INTEGER PRIMARY KEY,
    title TEXT NOT NULL,
    year INTEGER,
    formats_json TEXT,
    genres_json TEXT,
    styles_json TEXT,
    discogs_uri TEXT,
    date_added TEXT,
    cover_image_url TEXT,
    added_at TEXT NOT NULL,
    last_synced_at TEXT NOT NULL,
    removed_from_discogs_at TEXT
);

CREATE TABLE artists (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL
);

CREATE TABLE release_artists (
    release_id INTEGER NOT NULL REFERENCES releases(id) ON DELETE CASCADE,
    artist_id INTEGER NOT NULL REFERENCES artists(id) ON DELETE CASCADE,
    PRIMARY KEY (release_id, artist_id)
);

CREATE TABLE labels (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL
);

CREATE TABLE release_labels (
    release_id INTEGER NOT NULL REFERENCES releases(id) ON DELETE CASCADE,
    label_id INTEGER NOT NULL REFERENCES labels(id) ON DELETE CASCADE,
    catalog_number TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (release_id, label_id, catalog_number)
);

CREATE TABLE tracks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    release_id INTEGER NOT NULL REFERENCES releases(id) ON DELETE CASCADE,
    position TEXT,
    title TEXT NOT NULL,
    duration TEXT,
    artist TEXT
);

CREATE INDEX idx_tracks_release_id ON tracks(release_id);
CREATE INDEX idx_release_artists_artist_id ON release_artists(artist_id);
CREATE INDEX idx_release_labels_label_id ON release_labels(label_id);
