CREATE TABLE crates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE crate_releases (
    crate_id INTEGER NOT NULL REFERENCES crates(id) ON DELETE CASCADE,
    release_id INTEGER NOT NULL REFERENCES releases(id) ON DELETE CASCADE,
    added_at TEXT NOT NULL,
    PRIMARY KEY (crate_id, release_id)
);

CREATE INDEX idx_crate_releases_release_id ON crate_releases(release_id);
