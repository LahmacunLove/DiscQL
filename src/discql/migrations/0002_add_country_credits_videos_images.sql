ALTER TABLE releases ADD COLUMN country TEXT;
ALTER TABLE releases ADD COLUMN images_json TEXT;

CREATE TABLE release_videos (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    release_id INTEGER NOT NULL REFERENCES releases(id) ON DELETE CASCADE,
    title TEXT,
    url TEXT NOT NULL,
    duration INTEGER,
    description TEXT
);

CREATE INDEX idx_release_videos_release_id ON release_videos(release_id);

CREATE TABLE release_credits (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    release_id INTEGER NOT NULL REFERENCES releases(id) ON DELETE CASCADE,
    artist_id INTEGER NOT NULL REFERENCES artists(id) ON DELETE CASCADE,
    role TEXT,
    tracks TEXT,
    name_variation TEXT
);

CREATE INDEX idx_release_credits_release_id ON release_credits(release_id);
CREATE INDEX idx_release_credits_artist_id ON release_credits(artist_id);
