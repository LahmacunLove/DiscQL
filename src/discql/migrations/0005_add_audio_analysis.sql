ALTER TABLE tracks ADD COLUMN bpm REAL;
ALTER TABLE tracks ADD COLUMN musical_key TEXT;
ALTER TABLE tracks ADD COLUMN musical_key_scale TEXT;
ALTER TABLE tracks ADD COLUMN mood_json TEXT;
ALTER TABLE tracks ADD COLUMN mood_summary TEXT;
ALTER TABLE tracks ADD COLUMN waveform_path TEXT;
ALTER TABLE tracks ADD COLUMN analyzed_at TEXT;
