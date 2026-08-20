ALTER TABLE release_videos ADD COLUMN yt_title TEXT;
ALTER TABLE release_videos ADD COLUMN yt_duration INTEGER;
ALTER TABLE release_videos ADD COLUMN yt_uploader TEXT;
ALTER TABLE release_videos ADD COLUMN yt_description TEXT;
ALTER TABLE release_videos ADD COLUMN yt_tags_json TEXT;
ALTER TABLE release_videos ADD COLUMN yt_chapters_json TEXT;
ALTER TABLE release_videos ADD COLUMN yt_fetched_at TEXT;
ALTER TABLE release_videos ADD COLUMN video_type TEXT;
ALTER TABLE release_videos ADD COLUMN video_type_reasoning TEXT;
ALTER TABLE release_videos ADD COLUMN best_fuzzy_score REAL;
ALTER TABLE release_videos ADD COLUMN best_fuzzy_track_position TEXT;

CREATE TABLE track_video_matches (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    track_id INTEGER NOT NULL REFERENCES tracks(id) ON DELETE CASCADE,
    video_id INTEGER NOT NULL REFERENCES release_videos(id) ON DELETE CASCADE,
    fuzzy_score REAL NOT NULL,
    llm_confidence REAL,
    llm_reasoning TEXT,
    is_selected INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);

CREATE INDEX idx_track_video_matches_track_id ON track_video_matches(track_id);
CREATE INDEX idx_track_video_matches_video_id ON track_video_matches(video_id);
