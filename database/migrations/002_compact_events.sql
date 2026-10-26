-- Existing committed raw logs must be handled explicitly before enabling compact storage.
DO $$ BEGIN
    IF EXISTS (SELECT 1 FROM recommender.watch_events LIMIT 1) THEN
        RAISE EXCEPTION 'watch_events still contains data; export it before enabling compact events';
    END IF;
END $$;

CREATE TABLE recommender.event_video_keys (
    video_key integer PRIMARY KEY CHECK (video_key > 0),
    video_id text NOT NULL UNIQUE REFERENCES recommender.videos(id)
);

-- Integer dictionary codes avoid repeating 19-digit video IDs millions of times.
-- Paired arrays preserve each user's rounds, repeated watches and likes exactly.
CREATE TABLE recommender.user_event_history (
    user_id bigint PRIMARY KEY REFERENCES recommender.users(user_id),
    watch_video_keys integer[] NOT NULL,
    watch_rounds integer[] NOT NULL,
    like_video_keys integer[] NOT NULL,
    like_rounds integer[] NOT NULL,
    CHECK (cardinality(watch_video_keys) = cardinality(watch_rounds)),
    CHECK (cardinality(like_video_keys) = cardinality(like_rounds))
);

CREATE TABLE recommender.video_round_stats (
    video_key integer PRIMARY KEY REFERENCES recommender.event_video_keys(video_key),
    rounds integer[] NOT NULL,
    watch_counts integer[] NOT NULL,
    like_counts integer[] NOT NULL,
    CHECK (cardinality(rounds) = cardinality(watch_counts)),
    CHECK (cardinality(rounds) = cardinality(like_counts))
);

-- Published only after both history and statistics have committed together.
CREATE TABLE recommender.event_snapshot (
    singleton boolean PRIMARY KEY DEFAULT true CHECK (singleton),
    sha256 text NOT NULL,
    source_rows bigint NOT NULL CHECK (source_rows >= 0),
    unique_events bigint NOT NULL CHECK (unique_events >= 0),
    watch_count bigint NOT NULL CHECK (watch_count >= 0),
    like_count bigint NOT NULL CHECK (like_count >= 0),
    imported_at timestamptz NOT NULL DEFAULT now(),
    CHECK (unique_events = watch_count + like_count)
);

ALTER TABLE recommender.event_video_keys ENABLE ROW LEVEL SECURITY;
ALTER TABLE recommender.user_event_history ENABLE ROW LEVEL SECURITY;
ALTER TABLE recommender.video_round_stats ENABLE ROW LEVEL SECURITY;
ALTER TABLE recommender.event_snapshot ENABLE ROW LEVEL SECURITY;
REVOKE ALL ON recommender.event_video_keys, recommender.user_event_history,
    recommender.video_round_stats, recommender.event_snapshot FROM PUBLIC;
COMMENT ON TABLE recommender.watch_events IS
    'Legacy empty table. New event imports use user_event_history and video_round_stats.';
