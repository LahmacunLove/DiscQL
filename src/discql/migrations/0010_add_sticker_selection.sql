CREATE TABLE sticker_selection (
    release_id INTEGER PRIMARY KEY REFERENCES releases(id) ON DELETE CASCADE,
    added_at TEXT NOT NULL
);
