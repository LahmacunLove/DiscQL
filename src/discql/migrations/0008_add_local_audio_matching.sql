ALTER TABLE releases ADD COLUMN local_folder_path TEXT;
ALTER TABLE releases ADD COLUMN local_folder_match_score REAL;
ALTER TABLE releases ADD COLUMN local_matched_at TEXT;

ALTER TABLE tracks ADD COLUMN local_audio_path TEXT;
ALTER TABLE tracks ADD COLUMN local_match_score REAL;
